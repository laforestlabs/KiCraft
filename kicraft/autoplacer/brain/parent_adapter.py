"""Adapter between parent-level composition data and the unified PlacementSolver.

The parent compose pipeline holds:

  - ``LoadedSubcircuitArtifact`` instances (raw, untransformed children).
  - ``DerivedAttachmentConstraints`` from ``derive_attachment_constraints``.
  - parent-local ``Component`` instances (mounting holes, fiducials, etc.).

The unified ``PlacementSolver`` consumes:

  - a ``BoardState`` whose components carry ``block_blocker_set``,
    ``block_artifact_origin_offset``, ``block_side``, and
    ``allowed_rotations``.
  - a config dict whose ``component_zones`` may carry ``anchor_offset_mm``
    in addition to the existing ``edge``/``corner``/``zone`` keys.

This module bridges the two. Its functions are pure: they take parent-side
data and return solver-side data, with no side effects on disk or shared
state. PR4 (slate-cobalt-loom) lands this module dormant -- no callers are
wired up. PR5 replaces ``_compose_artifacts`` to use it.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .subcircuit_composer import (
    AttachmentConstraint,
    ChildArtifactPlacement,
    DerivedAttachmentConstraints,
    _compute_local_anchor_offset,
    dominant_blocker_side,
    extract_leaf_blocker_set,
    infer_interconnect_nets_local,
)
from .subcircuit_instances import (
    LoadedSubcircuitArtifact,
    transform_loaded_artifact,
)
from .types import (
    Component,
    Layer,
    Net,
    Point,
    SubCircuitDefinition,
)


_REF_SAFE_RE = re.compile(r"[^A-Za-z0-9]")


def synthetic_block_ref(child_index: int, sheet_name: str) -> str:
    """Stable ref string for the synthetic block representing one child."""
    safe = _REF_SAFE_RE.sub("_", sheet_name) or "child"
    return f"BLOCK_{child_index}_{safe}"


def _rotated(point: Point, rotation_deg: float) -> Point:
    rad = math.radians(rotation_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    return Point(
        point.x * cos_r - point.y * sin_r,
        point.x * sin_r + point.y * cos_r,
    )


def artifact_to_component(
    artifact: LoadedSubcircuitArtifact,
    *,
    ref: str,
    rotation: float = 0.0,
) -> Component:
    """Build a synthetic block ``Component`` carrying blocker-set metadata.

    The returned Component represents the artifact as a single rectangular
    block in the parent placer. ``block_blocker_set`` is populated, so
    pairs of blocks are checked for sparse-keepout compatibility before
    the solver decides whether to push them apart. The body bbox follows
    the artifact's local layout bbox.

    ``pos`` is initialized to the body center at the artifact's local
    origin (the solver will move it during placement). ``body_center``
    matches ``pos`` so courtyard math operates against the body as a
    whole rather than against any specific pad cluster.
    """
    layout = artifact.layout
    width, height = layout.bounding_box
    blocker_set = extract_leaf_blocker_set(artifact)
    side = dominant_blocker_side(blocker_set)

    body_center_local = Point(width / 2.0, height / 2.0)

    return Component(
        ref=ref,
        value=artifact.sheet_name,
        pos=Point(body_center_local.x, body_center_local.y),
        rotation=float(rotation),
        layer=Layer.FRONT,
        width_mm=float(width),
        height_mm=float(height),
        pads=[],
        kind="subcircuit",
        body_center=Point(body_center_local.x, body_center_local.y),
        block_blocker_set=blocker_set,
        block_artifact_origin_offset=Point(body_center_local.x, body_center_local.y),
        block_side=side,
    )


def infer_interconnect_nets_pre_placement(
    parent_def: SubCircuitDefinition,
    artifacts: list[LoadedSubcircuitArtifact],
    synthetic_refs_by_index: dict[int, str],
) -> dict[str, Net]:
    """Infer parent-level nets keyed off synthetic block refs.

    Wraps ``infer_interconnect_nets_local`` to translate
    ``(leaf_ref, pad_id)`` pad refs into ``(synthetic_block_ref, pad_id)``
    form. The unified solver uses these to compute attractive forces
    between blocks; routing is handled later by the parent stamp+route
    pipeline using the original leaf refs.
    """
    layouts_by_path = {a.instance_path: a.layout for a in artifacts}
    raw = infer_interconnect_nets_local(parent_def, layouts_by_path)

    leaf_to_block: dict[str, str] = {}
    for i, art in enumerate(artifacts):
        block_ref = synthetic_refs_by_index.get(i)
        if block_ref is None:
            continue
        for leaf_ref in art.layout.components:
            leaf_to_block[leaf_ref] = block_ref

    rewritten: dict[str, Net] = {}
    for name, net in raw.items():
        new_pad_refs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for leaf_ref, pad_id in net.pad_refs:
            block_ref = leaf_to_block.get(leaf_ref, leaf_ref)
            entry = (block_ref, pad_id)
            if entry in seen:
                continue
            seen.add(entry)
            new_pad_refs.append(entry)
        if len({ref for ref, _ in new_pad_refs}) < 2:
            continue
        rewritten[name] = Net(
            name=net.name,
            pad_refs=new_pad_refs,
            priority=net.priority,
            width_mm=net.width_mm,
            is_power=net.is_power,
        )
    return rewritten


def attachment_constraints_to_zones(
    derived: DerivedAttachmentConstraints,
    synthetic_refs_by_index: dict[int, str],
    artifacts: list[LoadedSubcircuitArtifact],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[float]]]:
    """Translate per-child attachment constraints into solver zone entries.

    Returns ``(zones, allowed_rotations)`` where:

      - ``zones[block_ref]`` carries a single ``edge``/``corner``/``zone``
        key (from the constraint), plus ``anchor_offset_mm`` (a ``Point``
        in local, unrotated frame: ``local_anchor - body_center``) and
        ``rotation`` (the chosen base rotation for this block).

      - ``allowed_rotations[block_ref]`` is the list of rotation
        candidates the solver may try via ``_optimize_rotations`` and
        the SA rotation move.

    ``anchor_offset_mm`` is computed in the *unrotated* local frame so
    the solver can rotate it per ``comp.rotation`` at placement time --
    this matches ``_world_artifact_origin``'s convention.
    """
    zones: dict[str, dict[str, Any]] = {}
    allowed_rots: dict[str, list[float]] = {}

    for child_index, spec in derived.child_specs.items():
        block_ref = synthetic_refs_by_index.get(child_index)
        if block_ref is None or not spec.constraints:
            continue
        if child_index >= len(artifacts):
            continue

        artifact = artifacts[child_index]
        layout = artifact.layout
        body_center = Point(layout.bounding_box[0] / 2.0, layout.bounding_box[1] / 2.0)

        # First strict constraint wins for the zone target. Edge/corner
        # constraints are strict by construction; zone constraints are
        # softer but still keyed by target.
        primary = next(
            (c for c in spec.constraints if c.strict),
            spec.constraints[0],
        )

        chosen_rotation = (
            spec.rotation_candidates[0] if spec.rotation_candidates else 0.0
        )

        # Compute local_anchor_offset at rotation 0 -- gives us the
        # unrotated local-frame anchor position. The solver rotates
        # per comp.rotation when placing.
        identity = transform_loaded_artifact(
            artifact, origin=Point(0.0, 0.0), rotation=0.0
        )
        blocker_set = extract_leaf_blocker_set(artifact)
        local_anchor = _compute_local_anchor_offset(
            identity,
            primary,
            spec.constraints,
            blocker_set,
            0.0,
        )
        anchor_offset = Point(
            local_anchor.x - body_center.x,
            local_anchor.y - body_center.y,
        )

        zone_entry: dict[str, Any] = {
            "anchor_offset_mm": anchor_offset,
            "rotation": float(chosen_rotation),
        }
        zone_entry[primary.target] = primary.value
        zones[block_ref] = zone_entry
        allowed_rots[block_ref] = list(spec.all_rotation_candidates)

    return zones, allowed_rots


def placements_from_solved_state(
    solved: dict[str, Component],
    artifacts: list[LoadedSubcircuitArtifact],
    synthetic_refs_by_index: dict[int, str],
) -> dict[str, ChildArtifactPlacement]:
    """Recover artifact-instance placements from solver output.

    The solver returns each block component positioned at its body
    center. This function inverts the
    ``world_origin = body_pos - rotated(body_center_offset, rotation)``
    relationship to recover the artifact's instance origin (which is
    what ``transform_loaded_artifact`` consumes downstream).
    """
    placements: dict[str, ChildArtifactPlacement] = {}
    for child_index, block_ref in synthetic_refs_by_index.items():
        comp = solved.get(block_ref)
        if comp is None or child_index >= len(artifacts):
            continue
        artifact = artifacts[child_index]
        layout = artifact.layout
        body_center_offset = Point(
            layout.bounding_box[0] / 2.0,
            layout.bounding_box[1] / 2.0,
        )
        rotated_offset = _rotated(body_center_offset, comp.rotation)
        artifact_origin = Point(
            comp.pos.x - rotated_offset.x,
            comp.pos.y - rotated_offset.y,
        )
        placements[artifact.instance_path] = ChildArtifactPlacement(
            artifact=artifact,
            origin=artifact_origin,
            rotation=float(comp.rotation),
        )
    return placements


__all__ = [
    "artifact_to_component",
    "attachment_constraints_to_zones",
    "infer_interconnect_nets_pre_placement",
    "placements_from_solved_state",
    "synthetic_block_ref",
]
