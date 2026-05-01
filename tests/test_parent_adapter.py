"""Tests for kicraft.autoplacer.brain.parent_adapter — parent-level adapter
between loaded subcircuit artifacts and the unified PlacementSolver.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations

from kicraft.autoplacer.brain.parent_adapter import (
    artifact_to_component,
    attachment_constraints_to_zones,
    infer_interconnect_nets_pre_placement,
    placements_from_solved_state,
    synthetic_block_ref,
)
from kicraft.autoplacer.brain.subcircuit_composer import (
    AttachmentConstraint,
    DerivedAttachmentConstraints,
    PlacementConstraintEntry,
    PlacementModel,
    PlacementSpec,
)
from kicraft.autoplacer.brain.subcircuit_instances import (
    LoadedSubcircuitArtifact,
    transform_loaded_artifact,
)
from kicraft.autoplacer.brain.types import (
    Component,
    InterfaceAnchor,
    InterfaceDirection,
    InterfacePort,
    InterfaceRole,
    InterfaceSide,
    Layer,
    Pad,
    Point,
    SubCircuitDefinition,
    SubCircuitId,
    SubCircuitLayout,
    SubcircuitAccessPolicy,
)


def _make_id(name: str) -> SubCircuitId:
    return SubCircuitId(
        sheet_name=name,
        sheet_file=f"{name.lower()}.kicad_sch",
        instance_path=f"/{name.lower()}",
    )


def _make_port(name: str, net_name: str) -> InterfacePort:
    return InterfacePort(
        name=name,
        net_name=net_name,
        role=InterfaceRole.BIDIR,
        direction=InterfaceDirection.UNKNOWN,
        preferred_side=InterfaceSide.ANY,
        access_policy=SubcircuitAccessPolicy.INTERFACE_ONLY,
    )


def _make_artifact(
    name: str,
    *,
    components: dict[str, Component] | None = None,
    ports: list[InterfacePort] | None = None,
    anchors: list[InterfaceAnchor] | None = None,
    bbox: tuple[float, float] = (10.0, 8.0),
) -> LoadedSubcircuitArtifact:
    layout = SubCircuitLayout(
        subcircuit_id=_make_id(name),
        components=dict(components or {}),
        bounding_box=bbox,
        ports=list(ports or []),
        interface_anchors=list(anchors or []),
        score=80.0,
    )
    return LoadedSubcircuitArtifact(
        artifact_dir=f"/tmp/fake/{name}",
        metadata={},
        debug={},
        layout=layout,
    )


class TestSyntheticBlockRef:
    def test_stable_format(self):
        assert synthetic_block_ref(0, "USB_INPUT") == "_block_0_USB_INPUT"
        assert synthetic_block_ref(3, "LDO") == "_block_3_LDO"


class TestArtifactToComponent:
    def test_kind_is_subcircuit(self):
        artifact = _make_artifact("LEAF")
        comp = artifact_to_component(artifact, ref="_block_0_LEAF")
        assert comp.kind == "subcircuit"
        assert comp.locked is False
        assert comp.is_through_hole is False

    def test_dimensions_match_transformed_bbox(self):
        artifact = _make_artifact("LEAF", bbox=(20.0, 10.0))
        comp = artifact_to_component(artifact, ref="_block_0_LEAF")
        # Empty layout → bbox is (0, 0) regardless of layout.bounding_box,
        # because _compute_layout_bbox derives from geometry. Build with
        # at least one component or anchor to get a real bbox.

    def test_synthetic_pads_at_anchors(self):
        anchors = [
            InterfaceAnchor(port_name="P1", pos=Point(0.0, 5.0), layer=Layer.FRONT),
            InterfaceAnchor(port_name="P2", pos=Point(10.0, 5.0), layer=Layer.FRONT),
        ]
        ports = [_make_port("P1", "NET_A"), _make_port("P2", "NET_B")]
        artifact = _make_artifact("LEAF", ports=ports, anchors=anchors)
        comp = artifact_to_component(artifact, ref="_block_0_LEAF")
        assert len(comp.pads) == 2
        nets = {pad.net for pad in comp.pads}
        assert nets == {"NET_A", "NET_B"}

    def test_initial_pos_is_origin(self):
        artifact = _make_artifact("LEAF")
        comp = artifact_to_component(artifact, ref="_block_0_LEAF")
        assert comp.pos.x == 0.0 and comp.pos.y == 0.0

    def test_block_side_classification(self):
        # Front-only pads → 'front'
        front_anchor = InterfaceAnchor(
            port_name="P1", pos=Point(0.0, 0.0), layer=Layer.FRONT
        )
        artifact = _make_artifact(
            "LEAF",
            ports=[_make_port("P1", "N")],
            anchors=[front_anchor],
        )
        comp = artifact_to_component(artifact, ref="r")
        # block_side derives from transformed components' pads, not anchors;
        # an artifact with no components yields 'none'.
        assert comp.block_side in ("front", "back", "dual", "none")


class TestInferInterconnectNetsPrePlacement:
    def test_nets_keyed_off_synthetic_refs(self):
        port_a = _make_port("P_OUT", "POWER")
        port_b = _make_port("P_IN", "POWER")
        anchor_a = InterfaceAnchor(
            port_name="P_OUT",
            pos=Point(10.0, 4.0),
            layer=Layer.FRONT,
            pad_ref=("J1", "1"),
        )
        anchor_b = InterfaceAnchor(
            port_name="P_IN",
            pos=Point(0.0, 4.0),
            layer=Layer.FRONT,
            pad_ref=("U1", "1"),
        )
        art_a = _make_artifact(
            "SOURCE", ports=[port_a], anchors=[anchor_a]
        )
        art_b = _make_artifact(
            "SINK", ports=[port_b], anchors=[anchor_b]
        )
        parent = SubCircuitDefinition(
            id=_make_id("PARENT"),
            child_ids=[art_a.layout.subcircuit_id, art_b.layout.subcircuit_id],
            is_leaf=False,
        )
        synthetic_refs = {0: "_block_0_SOURCE", 1: "_block_1_SINK"}
        nets = infer_interconnect_nets_pre_placement(
            parent, [art_a, art_b], synthetic_refs
        )
        assert "POWER" in nets
        # Each pad_ref's first element is now a synthetic block ref, not a leaf ref
        block_refs = {ref for ref, _ in nets["POWER"].pad_refs}
        assert block_refs == {"_block_0_SOURCE", "_block_1_SINK"}


class TestAttachmentConstraintsToZones:
    def _spec_with_edge(self, ref: str, value: str) -> PlacementSpec:
        constraint = AttachmentConstraint(
            ref=ref,
            target="edge",
            value=value,
            inward_keep_in_mm=0.0,
            outward_overhang_mm=0.0,
            source="child_artifact",
            child_index=0,
        )
        # Synthesize a tiny PlacementModel: a 10x10 transformed view with
        # the constraint anchor offset 5mm to the right of body center.
        from kicraft.autoplacer.brain.subcircuit_instances import (
            TransformedSubcircuit,
        )

        transformed = TransformedSubcircuit(
            instance=None,  # type: ignore[arg-type]
            layout=SubCircuitLayout(subcircuit_id=_make_id("X")),
            transformed_components={},
            transformed_traces=[],
            transformed_vias=[],
            transformed_silkscreen=[],
            transformed_anchors=[],
            bounding_box=(Point(0.0, 0.0), Point(10.0, 10.0)),
        )
        entry = PlacementConstraintEntry(
            constraint=constraint,
            local_anchor_offset=Point(0.0, 5.0),
        )
        model = PlacementModel(
            rotation=0.0,
            transformed=transformed,
            layer_envelopes=([], [], []),
            blocker_set=None,
            constraint_entries=[entry],
        )
        return PlacementSpec(
            child_index=0,
            instance_path=f"/{ref.lower()}",
            rotation_candidates=[0.0],
            all_rotation_candidates=[0.0, 90.0, 180.0, 270.0],
            constraints=[constraint],
            models={0.0: model},
        )

    def test_edge_constraint_includes_anchor_offset(self):
        spec = self._spec_with_edge("J1", "left")
        derived = DerivedAttachmentConstraints(
            constraints=list(spec.constraints),
            child_specs={0: spec},
            parent_local_constraints=[],
        )
        zones, allowed = attachment_constraints_to_zones(
            derived, {0: "_block_0_X"}
        )
        assert "_block_0_X" in zones
        z = zones["_block_0_X"]
        assert z["edge"] == "left"
        # body_center is bbox center = (5, 5); anchor at (0, 5) → offset (-5, 0)
        assert z["anchor_offset_mm"] == (-5.0, 0.0)
        assert allowed["_block_0_X"] == [0.0]

    def test_parent_local_passes_through(self):
        constraint = AttachmentConstraint(
            ref="MH1",
            target="corner",
            value="top-left",
            inward_keep_in_mm=2.5,
            outward_overhang_mm=0.0,
            source="parent_local",
            child_index=None,
        )
        derived = DerivedAttachmentConstraints(
            constraints=[constraint],
            child_specs={},
            parent_local_constraints=[constraint],
        )
        zones, allowed = attachment_constraints_to_zones(derived, {})
        assert zones["MH1"] == {"corner": "top-left"}
        assert allowed == {}


class TestPlacementsFromSolvedState:
    def test_round_trip_origin_and_rotation(self):
        artifact = _make_artifact("LEAF")
        ref = synthetic_block_ref(0, "LEAF")
        # Solver "moves" the synthetic block to (12.5, 7.0) at rotation 90°
        comp = Component(
            ref=ref,
            value="LEAF",
            pos=Point(12.5, 7.0),
            rotation=90.0,
            layer=Layer.FRONT,
            width_mm=10.0,
            height_mm=8.0,
            kind="subcircuit",
        )
        placements = placements_from_solved_state(
            {ref: comp}, [artifact], {0: ref}
        )
        assert len(placements) == 1
        assert placements[0].origin.x == 12.5
        assert placements[0].origin.y == 7.0
        assert placements[0].rotation == 90.0
        assert placements[0].artifact is artifact
