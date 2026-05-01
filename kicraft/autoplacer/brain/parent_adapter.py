"""Parent-level adapter for the unified PlacementSolver.

Bridges loaded subcircuit artifacts to the synthetic ``Component`` form
that ``PlacementSolver.solve()`` consumes. The pre-placement and
post-placement helpers here let the same solver run leaf and parent
levels — see ``kicraft/cli/compose_subcircuits.py::_compose_artifacts``
for the cutover call site.
"""

from __future__ import annotations

from .subcircuit_composer import (
    AttachmentConstraint,
    ChildArtifactPlacement,
    DerivedAttachmentConstraints,
    PlacementSpec,
    _append_pad_ref,
    _dedupe_inferred_nets,
    _resolve_layout_port_pad_ref,
)
from .subcircuit_instances import (
    LoadedSubcircuitArtifact,
    transform_loaded_artifact,
)
from .types import (
    Component,
    Layer,
    Net,
    Pad,
    Point,
    SubCircuitDefinition,
)


def synthetic_block_ref(index: int, sheet_name: str) -> str:
    """Stable synthetic ref for a block component."""
    return f"_block_{index}_{sheet_name}"


def _classify_block_side(transformed) -> str:
    """Infer the dominant copper side of an artifact for opposite-side packing.

    'front' / 'back' / 'dual' / 'none' based on which layers carry pad
    copper. A child whose pads land mostly on one side stacks well with a
    child of the opposite side; 'dual' children pack reasonably either way.
    """
    front = back = 0
    for comp in transformed.transformed_components.values():
        for pad in comp.pads:
            if pad.layer == Layer.BACK:
                back += 1
            else:
                front += 1
    if front == 0 and back == 0:
        return "none"
    if front == 0:
        return "back"
    if back == 0:
        return "front"
    ratio = max(front, back) / (front + back)
    return "dual" if ratio < 0.85 else ("front" if front >= back else "back")


def artifact_to_component(
    artifact: LoadedSubcircuitArtifact,
    *,
    ref: str,
    rotation: float = 0.0,
) -> Component:
    """Wrap a loaded artifact as a synthetic ``Component`` for the solver.

    The synthetic block carries:
      - bbox dimensions = artifact bbox at the chosen rotation
      - synthetic pads at each interface anchor position (with the
        anchor's net) so the solver's connectivity graph attracts blocks
      - ``kind="subcircuit"`` so the solver gates leaf-only steps off
      - ``block_side`` derived from pad-layer distribution for the
        parent-level opposite-side packing reward
    """
    transformed = transform_loaded_artifact(artifact, origin=Point(0.0, 0.0), rotation=rotation)
    bbox_min, bbox_max = transformed.bounding_box
    width = bbox_max.x - bbox_min.x
    height = bbox_max.y - bbox_min.y
    bbox_center = Point(
        (bbox_min.x + bbox_max.x) / 2.0,
        (bbox_min.y + bbox_max.y) / 2.0,
    )

    # Use bbox center as both pos and body_center so the solver's
    # half-extent / legalize logic sees a tight bbox. The artifact's
    # local-frame origin is recovered downstream by subtracting
    # bbox_center from comp.pos in placements_from_solved_state.
    pads: list[Pad] = []
    for anchor in transformed.transformed_anchors:
        if not anchor.port_name:
            continue
        pad_id = anchor.pad_ref[1] if anchor.pad_ref else anchor.port_name
        net_name = ""
        for port in artifact.layout.ports:
            if port.name == anchor.port_name:
                net_name = port.net_name
                break
        pads.append(
            Pad(
                ref=ref,
                pad_id=pad_id,
                pos=Point(anchor.pos.x, anchor.pos.y),
                layer=anchor.layer,
                net=net_name,
            )
        )

    return Component(
        ref=ref,
        value=artifact.sheet_name,
        pos=Point(bbox_center.x, bbox_center.y),
        rotation=rotation,
        layer=Layer.FRONT,
        width_mm=width,
        height_mm=height,
        pads=pads,
        locked=False,
        kind="subcircuit",
        is_through_hole=False,
        body_center=Point(bbox_center.x, bbox_center.y),
        block_side=_classify_block_side(transformed),
    )


def infer_interconnect_nets_pre_placement(
    parent_definition: SubCircuitDefinition,
    loaded_artifacts: list[LoadedSubcircuitArtifact],
    synthetic_refs_by_index: dict[int, str],
) -> dict[str, Net]:
    """Pre-placement parent interconnect nets keyed off synthetic block refs.

    Walks each artifact's local-frame ports + anchors and emits ``Net``
    objects whose pad_refs point at the *synthetic block component* (not
    the leaf-level ref the post-placement variant uses). This way the
    solver's connectivity graph attracts whole blocks, which is the
    correct semantics at parent level.
    """
    inferred: dict[str, Net] = {}
    artifact_by_path = {a.instance_path: a for a in loaded_artifacts}
    index_by_path = {
        a.instance_path: i for i, a in enumerate(loaded_artifacts)
    }

    for child_id in parent_definition.child_ids:
        artifact = artifact_by_path.get(child_id.instance_path)
        if artifact is None:
            continue
        index = index_by_path[artifact.instance_path]
        synthetic_ref = synthetic_refs_by_index.get(index)
        if synthetic_ref is None:
            continue

        layout = artifact.layout
        anchors_by_port = {
            anchor.port_name: anchor for anchor in layout.interface_anchors
        }
        center = Point(layout.width / 2.0, layout.height / 2.0)

        for port in layout.ports:
            if not port.net_name:
                continue
            leaf_pad_ref = _resolve_layout_port_pad_ref(
                anchors_by_port,
                layout.components,
                center,
                port.name,
                port.net_name,
            )
            if leaf_pad_ref is None:
                continue
            # Rewrite the pad ref onto the synthetic block component so
            # the connectivity graph at parent level sees blocks, not
            # internal leaf components.
            block_pad_id = leaf_pad_ref[1]
            _append_pad_ref(
                inferred,
                port.net_name,
                (synthetic_ref, block_pad_id),
            )

    return _dedupe_inferred_nets(inferred)


def attachment_constraints_to_zones(
    derived: DerivedAttachmentConstraints,
    synthetic_refs_by_index: dict[int, str],
) -> tuple[dict[str, dict], dict[str, list[float]]]:
    """Map ``AttachmentConstraint`` instances onto the solver's zone schema.

    Returns:
      - ``zones``: dict[ref -> {edge|corner|zone: value, anchor_offset_mm: (ax, ay)}]
        keyed by synthetic block ref for child constraints, or by the
        parent-local component ref directly. ``anchor_offset_mm`` is
        block-relative (offset from body_center to the anchor) so the
        solver pins (body_center + anchor_offset) on the edge/corner.
      - ``allowed_rotations``: dict[synthetic_ref -> [rotation, ...]]
        carrying the constraint's allowed rotation candidates. The
        solver consults this in ``_optimize_rotations`` and the SA
        rotation move.
    """
    zones: dict[str, dict] = {}
    allowed_rotations: dict[str, list[float]] = {}

    for child_index, spec in derived.child_specs.items():
        synthetic_ref = synthetic_refs_by_index.get(child_index)
        if synthetic_ref is None:
            continue
        # Lock to the first rotation candidate; constrained blocks pick
        # one rotation up front (matching the old loop's "first feasible
        # rotation wins" behavior). The solver may still try alternates
        # via allowed_rotations for unconstrained-axis perturbation.
        chosen_rotation = spec.rotation_candidates[0]
        model = spec.models[chosen_rotation]
        bbox_min, bbox_max = model.transformed.bounding_box
        body_center = Point(
            (bbox_min.x + bbox_max.x) / 2.0,
            (bbox_min.y + bbox_max.y) / 2.0,
        )

        primary_entry = next(
            (
                entry
                for entry in model.constraint_entries
                if entry.constraint.target in ("edge", "corner")
            ),
            None,
        )
        if primary_entry is None:
            continue

        anchor_offset = (
            primary_entry.local_anchor_offset.x - body_center.x,
            primary_entry.local_anchor_offset.y - body_center.y,
        )
        zone_entry: dict = {
            primary_entry.constraint.target: primary_entry.constraint.value,
            "anchor_offset_mm": anchor_offset,
            "rotation": chosen_rotation,
        }
        zones[synthetic_ref] = zone_entry
        allowed_rotations[synthetic_ref] = list(spec.rotation_candidates)

    for constraint in derived.parent_local_constraints:
        if constraint.target not in ("edge", "corner", "zone"):
            continue
        zones[constraint.ref] = {constraint.target: constraint.value}

    return zones, allowed_rotations


def placements_from_solved_state(
    solved: dict[str, Component],
    loaded_artifacts: list[LoadedSubcircuitArtifact],
    synthetic_refs_by_index: dict[int, str],
) -> list[ChildArtifactPlacement]:
    """Read solver-output position/rotation back into ``ChildArtifactPlacement``.

    The synthetic block uses bbox-center as both ``pos`` and ``body_center``,
    so the artifact instance origin (artifact local (0, 0) in parent
    coordinates) is recovered by subtracting the local-frame bbox center
    from ``comp.pos``.
    """
    placements: list[ChildArtifactPlacement] = []
    for index, artifact in enumerate(loaded_artifacts):
        synthetic_ref = synthetic_refs_by_index[index]
        comp = solved[synthetic_ref]
        # Recompute the local-frame bbox center at the chosen rotation
        # to undo the bbox-center offset baked into comp.pos.
        local_transformed = transform_loaded_artifact(
            artifact, origin=Point(0.0, 0.0), rotation=comp.rotation
        )
        lbb_min, lbb_max = local_transformed.bounding_box
        offset_x = (lbb_min.x + lbb_max.x) / 2.0
        offset_y = (lbb_min.y + lbb_max.y) / 2.0
        placements.append(
            ChildArtifactPlacement(
                artifact=artifact,
                origin=Point(comp.pos.x - offset_x, comp.pos.y - offset_y),
                rotation=comp.rotation,
            )
        )
    return placements
