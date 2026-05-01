"""End-to-end smoke test for the unified parent placer.

Builds a minimal 3-block synthetic project (no real KiCad files) and
runs the new ``_compose_artifacts`` adapter through the unified
``PlacementSolver`` path. Asserts that solver output produces a valid
``ParentCompositionState`` with no overlaps and all blocks within the
final outline.
"""

from __future__ import annotations

import pytest

from kicraft.cli.compose_subcircuits import _compose_artifacts
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


def _id(name: str) -> SubCircuitId:
    return SubCircuitId(
        sheet_name=name,
        sheet_file=f"{name.lower()}.kicad_sch",
        instance_path=f"/{name.lower()}",
    )


def _port(name: str, net: str) -> InterfacePort:
    return InterfacePort(
        name=name,
        net_name=net,
        role=InterfaceRole.BIDIR,
        direction=InterfaceDirection.UNKNOWN,
        preferred_side=InterfaceSide.ANY,
        access_policy=SubcircuitAccessPolicy.INTERFACE_ONLY,
    )


def _pad(ref: str, pad_id: str, x: float, y: float, net: str) -> Pad:
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net=net, layer=Layer.FRONT)


def _component(ref: str, x: float, y: float, w: float, h: float, *, pads=()) -> Component:
    return Component(
        ref=ref,
        value=ref,
        pos=Point(x, y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=w,
        height_mm=h,
        body_center=Point(x, y),
        pads=list(pads),
    )


def _artifact(name: str, components: dict[str, Component], anchors, ports, bbox) -> LoadedSubcircuitArtifact:
    layout = SubCircuitLayout(
        subcircuit_id=_id(name),
        components=components,
        bounding_box=bbox,
        ports=ports,
        interface_anchors=anchors,
        score=80.0,
    )
    return LoadedSubcircuitArtifact(
        artifact_dir=f"/tmp/fake/{name}",
        metadata={},
        debug={},
        layout=layout,
    )


def _bboxes_overlap(a, b, tol=0.05):
    """Check meaningful overlap with sub-mm tolerance.

    Sub-tol overlaps (< 50 µm) are below the placement clearance
    granularity and aren't real DRC violations; we treat them as
    non-overlapping.
    """
    return not (
        a[1].x <= b[0].x + tol
        or b[1].x <= a[0].x + tol
        or a[1].y <= b[0].y + tol
        or b[1].y <= a[0].y + tol
    )


@pytest.fixture
def three_block_artifacts():
    """Three small synthetic leaves with one shared net (POWER) connecting all."""
    artifacts = []
    for i, name in enumerate(["SOURCE", "MIDDLE", "SINK"]):
        comp = _component(
            ref=f"R{i+1}",
            x=2.5,
            y=2.5,
            w=5.0,
            h=5.0,
            pads=[_pad(f"R{i+1}", "1", 0.0, 2.5, "POWER")],
        )
        anchors = [
            InterfaceAnchor(
                port_name="POWER",
                pos=Point(0.0, 2.5),
                layer=Layer.FRONT,
                pad_ref=(f"R{i+1}", "1"),
            )
        ]
        ports = [_port("POWER", "POWER")]
        artifacts.append(
            _artifact(name, {f"R{i+1}": comp}, anchors, ports, bbox=(5.0, 5.0))
        )
    return artifacts


@pytest.fixture
def parent_definition(three_block_artifacts):
    return SubCircuitDefinition(
        id=_id("PARENT"),
        child_ids=[a.layout.subcircuit_id for a in three_block_artifacts],
        is_leaf=False,
    )


class TestUnifiedCompose:
    def test_returns_valid_state(self, three_block_artifacts, parent_definition):
        state, payloads = _compose_artifacts(
            three_block_artifacts,
            spacing_mm=2.0,
            rotation_step_deg=0.0,
            parent_definition=parent_definition,
            seed=1,
        )
        assert state is not None
        assert len(state.entries) == 3
        assert len(payloads) == 3

    def _world_bboxes(self, artifacts, state):
        """Recompute per-block world bboxes from the transformed artifact.

        ``entry.transformed_bbox`` is just (rotated_w, rotated_h); for
        rotated blocks the local-frame (0, 0) — i.e. ``entry.origin`` —
        is not a corner of the world bbox, so summing them is wrong.
        Use ``transform_loaded_artifact`` to get the actual world bbox.
        """
        bboxes = []
        for art, entry in zip(artifacts, state.entries):
            t = transform_loaded_artifact(art, origin=entry.origin, rotation=entry.rotation)
            bboxes.append(t.bounding_box)
        return bboxes

    def test_no_block_overlap(self, three_block_artifacts, parent_definition):
        state, _ = _compose_artifacts(
            three_block_artifacts,
            spacing_mm=2.0,
            rotation_step_deg=0.0,
            parent_definition=parent_definition,
            seed=2,
        )
        bboxes = self._world_bboxes(three_block_artifacts, state)
        for i, a in enumerate(bboxes):
            for b in bboxes[i + 1 :]:
                assert not _bboxes_overlap(a, b), f"blocks overlap: {a} vs {b}"

    def test_blocks_inside_outline(self, three_block_artifacts, parent_definition):
        state, _ = _compose_artifacts(
            three_block_artifacts,
            spacing_mm=2.0,
            rotation_step_deg=0.0,
            parent_definition=parent_definition,
            seed=3,
        )
        tl, br = state.bounding_box
        bboxes = self._world_bboxes(three_block_artifacts, state)
        for bbox in bboxes:
            assert bbox[0].x >= tl.x - 1e-3
            assert bbox[0].y >= tl.y - 1e-3
            assert bbox[1].x <= br.x + 1e-3
            assert bbox[1].y <= br.y + 1e-3

    def test_interconnect_nets_inferred(self, three_block_artifacts, parent_definition):
        state, _ = _compose_artifacts(
            three_block_artifacts,
            spacing_mm=2.0,
            rotation_step_deg=0.0,
            parent_definition=parent_definition,
            seed=4,
        )
        # The shared POWER net wires all three blocks together
        assert state.inferred_interconnect_net_count >= 1

    def test_seed_determinism(self, three_block_artifacts, parent_definition):
        state_a, _ = _compose_artifacts(
            three_block_artifacts,
            spacing_mm=2.0,
            rotation_step_deg=0.0,
            parent_definition=parent_definition,
            seed=42,
        )
        state_b, _ = _compose_artifacts(
            three_block_artifacts,
            spacing_mm=2.0,
            rotation_step_deg=0.0,
            parent_definition=parent_definition,
            seed=42,
        )
        for ea, eb in zip(state_a.entries, state_b.entries):
            assert ea.origin.x == pytest.approx(eb.origin.x)
            assert ea.origin.y == pytest.approx(eb.origin.y)
            assert ea.rotation == pytest.approx(eb.rotation)
