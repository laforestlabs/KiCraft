"""Solver/compose internals for identical-leaf replica orientation coherence.

Slice 2 of the board-quality contracts plan: a replica's leaf geometry is
already a rigid copy of its donor's, but the parent rotation search used to
optimize each synthetic block independently, so two copies of one circuit
could ship at different parent angles. These tests cover the mechanism:

  - the greedy block rotation search and the SA refinement treat a coupled
    group as one move (jointly scored, jointly restored) -- the SA moves live
    in ``tests/test_sa_refine.py`` and the end-to-end solve in
    ``tests/test_stack_compatible_blocks.py``;
  - ``_set_block_rotation`` applies measured per-rotation extents (never a
    transpose guess) and errors on missing geometry;
  - the empty rotation intersection and the replica-group boundaries
    (missing donor, self-donor, cycle) are hard errors;
  - ``_validate_replica_rotations`` hard-rejects a mismatched recovered
    placement, and ``_compose_artifacts`` calls it.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.placement_solver import (
    PlacementSolver,
    _set_block_rotation,
)
from kicraft.autoplacer.brain.subcircuit_composer import ChildArtifactPlacement
from kicraft.autoplacer.brain.subcircuit_instances import (
    LoadedSubcircuitArtifact,
)
from kicraft.autoplacer.brain.types import (
    BlockRotationGeometry,
    BoardState,
    Component,
    Layer,
    Pad,
    Point,
    SubCircuitId,
    SubCircuitLayout,
)
from kicraft.cli import compose_subcircuits as compose_module
from kicraft.cli.compose_subcircuits import (
    _compose_artifacts,
    _replica_rotation_groups,
    _validate_replica_rotations,
)


# ---------------------------------------------------------------------------
# Helpers


def _id(name: str) -> SubCircuitId:
    return SubCircuitId(
        sheet_name=name,
        sheet_file=f"{name.lower()}.kicad_sch",
        instance_path=f"/{name.lower()}",
    )


def _pad(ref: str, pad_id: str, x: float, y: float) -> Pad:
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net="", layer=Layer.FRONT)


def _artifact(
    name: str,
    components: dict[str, Component],
    width: float,
    height: float,
    *,
    replicated_from: str | None = None,
) -> LoadedSubcircuitArtifact:
    layout = SubCircuitLayout(
        subcircuit_id=_id(name),
        components=components,
        traces=[],
        vias=[],
        bounding_box=(width, height),
        ports=[],
        interface_anchors=[],
        score=75.0,
        replicated_from=replicated_from,
    )
    return LoadedSubcircuitArtifact(
        artifact_dir=f"/fake/{name}",
        metadata={},
        debug={},
        layout=layout,
        source_files={},
    )


def _driver_leaf(prefix: str) -> dict[str, Component]:
    """Identical two-part leaf body; only the refs differ between copies."""
    return {
        f"{prefix}1": Component(
            ref=f"{prefix}1",
            value="DRV8833",
            pos=Point(5.0, 8.0),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=10.0,
            height_mm=16.0,
            kind="ic",
            pads=[
                _pad(f"{prefix}1", "1", 1.0, 8.0),
                _pad(f"{prefix}1", "2", 9.0, 8.0),
            ],
        ),
        f"{prefix}2": Component(
            ref=f"{prefix}2",
            value="100n",
            pos=Point(14.0, 4.0),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=2.0,
            height_mm=1.0,
            kind="passive",
            pads=[
                _pad(f"{prefix}2", "1", 13.5, 4.0),
                _pad(f"{prefix}2", "2", 14.5, 4.0),
            ],
        ),
    }


def _driver_artifacts() -> tuple[LoadedSubcircuitArtifact, LoadedSubcircuitArtifact]:
    donor = _artifact("DRIVER_A", _driver_leaf("U"), 20.0, 20.0)
    replica = _artifact(
        "DRIVER_B", _driver_leaf("R"), 20.0, 20.0, replicated_from="/driver_a"
    )
    return donor, replica


def _wide_neighbour() -> LoadedSubcircuitArtifact:
    """Asymmetric neighbour so block rotation is a real choice, not a no-op."""
    return _artifact(
        "POWER",
        {
            "U9": Component(
                ref="U9",
                value="BUCK",
                pos=Point(6.0, 3.0),
                rotation=0.0,
                layer=Layer.FRONT,
                width_mm=12.0,
                height_mm=4.0,
                kind="ic",
                pads=[_pad("U9", "1", 1.0, 3.0), _pad("U9", "2", 11.0, 3.0)],
            )
        },
        20.0,
        8.0,
    )


def _synthetic_block(
    ref: str,
    *,
    pos: Point = Point(20.0, 20.0),
    dims: dict[float, tuple[float, float]],
    rotation: float = 0.0,
    donor: str | None = None,
    allowed: list[float] | None = None,
) -> Component:
    width, height = dims[rotation]
    comp = Component(
        ref=ref,
        value=ref,
        pos=Point(pos.x, pos.y),
        rotation=rotation,
        layer=Layer.FRONT,
        width_mm=width,
        height_mm=height,
        kind="subcircuit",
        body_center=Point(pos.x, pos.y),
        block_artifact_origin_offset=Point(width / 2.0, height / 2.0),
        block_rotation_geometry={
            rot: BlockRotationGeometry(width_mm=w, height_mm=h)
            for rot, (w, h) in dims.items()
        },
    )
    if allowed is not None:
        comp.allowed_rotations = list(allowed)
    comp.block_replication_donor = donor
    return comp


def _board(comps: dict[str, Component]) -> BoardState:
    return BoardState(
        components=comps,
        nets={},
        traces=[],
        vias=[],
        silkscreen=[],
        board_outline=(Point(0.0, 0.0), Point(120.0, 120.0)),
    )


# ---------------------------------------------------------------------------
# Measured rotation extents


def test_set_block_rotation_uses_measured_extents_not_transpose():
    """90° extents come from the leaf's real bbox, not a transpose guess."""
    comp = _synthetic_block(
        "BLOCK_0",
        dims={0.0: (10.0, 3.0), 90.0: (7.0, 3.0), 180.0: (10.0, 3.0), 270.0: (7.0, 3.0)},
    )
    _set_block_rotation(comp, 90.0)
    assert comp.rotation == 90.0
    assert (comp.width_mm, comp.height_mm) == (7.0, 3.0)  # NOT (3.0, 10.0)


def test_set_block_rotation_missing_geometry_is_error_and_leaves_extents():
    """No geometry entry = error; never a silent transpose fallback."""
    comp = _synthetic_block(
        "BLOCK_0",
        dims={0.0: (10.0, 3.0), 90.0: (7.0, 3.0)},
    )
    with pytest.raises(ValueError, match=r"missing_block_rotation_geometry:BLOCK_0@45"):
        _set_block_rotation(comp, 45.0)
    assert (comp.rotation, comp.width_mm, comp.height_mm) == (0.0, 10.0, 3.0)


def test_empty_rotation_intersection_raises_replica_rotation_conflict():
    """A group whose members share no legal angle is a hard error."""
    comps = {
        "A": _synthetic_block(
            "A", dims={0.0: (10.0, 3.0), 90.0: (3.0, 10.0)}, allowed=[0.0, 90.0]
        ),
        "B": _synthetic_block(
            "B",
            dims={180.0: (10.0, 3.0), 270.0: (3.0, 10.0)},
            rotation=180.0,
            allowed=[180.0, 270.0],
        ),
    }
    comps["B"].block_replication_donor = "A"
    solver = PlacementSolver(_board(comps), config={}, seed=0)
    solver._pinned_targets = {}
    with pytest.raises(ValueError, match=r"^replica_rotation_conflict:A$"):
        solver._optimize_rotations(comps, solver.state)


# ---------------------------------------------------------------------------
# Joint greedy scoring


def test_greedy_group_rotation_scores_every_member_together():
    """The winner is chosen with ALL members rotated, not donor-then-copy.

    The scorer only rewards 90° when BOTH blocks are at 90°. A search that
    scored the donor alone (sibling still at 0°) could never see 1.0, so the
    group would stay at its initial angle.
    """
    dims = {0.0: (4.0, 2.0), 90.0: (2.0, 4.0), 180.0: (4.0, 2.0), 270.0: (2.0, 4.0)}
    comps = {
        "BLOCK_A": _synthetic_block("BLOCK_A", dims=dims),
        "BLOCK_B": _synthetic_block("BLOCK_B", dims=dims, donor="BLOCK_A"),
    }
    solver = PlacementSolver(_board(comps), config={}, seed=0)
    solver._pinned_targets = {}

    seen_group_moves = []

    def _joint_score(_state):
        angles = {float(comps[ref].rotation) for ref in ("BLOCK_A", "BLOCK_B")}
        if len(angles) == 1 and angles != {0.0}:
            seen_group_moves.append(next(iter(angles)))
        return 1.0 if angles == {90.0} else 0.0

    solver._score_rotation_for_block = _joint_score

    solver._optimize_rotations(comps, solver.state)

    assert seen_group_moves
    for ref in ("BLOCK_A", "BLOCK_B"):
        assert comps[ref].rotation == 90.0
        assert (comps[ref].width_mm, comps[ref].height_mm) == (2.0, 4.0)


def test_greedy_group_prefers_initial_angle_on_ties_and_restores_all_members():
    """On a tie the group keeps its initial angle and every extents snapshot."""
    dims = {0.0: (4.0, 2.0), 90.0: (2.0, 4.0), 180.0: (8.0, 6.0), 270.0: (6.0, 8.0)}
    comps = {
        "BLOCK_A": _synthetic_block("BLOCK_A", dims=dims),
        "BLOCK_B": _synthetic_block("BLOCK_B", dims=dims, donor="BLOCK_A"),
    }
    solver = PlacementSolver(_board(comps), config={}, seed=0)
    solver._pinned_targets = {}
    # Constant score => every candidate ties the initial angle.
    solver._score_rotation_for_block = lambda _state: 5.0

    solver._optimize_rotations(comps, solver.state)

    for ref in ("BLOCK_A", "BLOCK_B"):
        assert comps[ref].rotation == 0.0
        assert (comps[ref].width_mm, comps[ref].height_mm) == (4.0, 2.0)


# ---------------------------------------------------------------------------
# Group resolution boundaries


def test_missing_donor_subset_forms_no_group():
    replica = _artifact(
        "ORPHAN",
        _driver_leaf("U"),
        20.0,
        20.0,
        replicated_from="/not_loaded",
    )
    neighbour = _wide_neighbour()

    state, _ = _compose_artifacts(
        [replica, neighbour],
        spacing_mm=2.0,
        rotation_step_deg=0.0,
        parent_definition=None,
        pcb_path=None,
        cfg={},
        seed=0,
    )
    assert state.packing_metadata["replica_rotation_groups"] == []
    assert len(state.entries) == 2


def test_manual_layout_does_not_acquire_orientation_coupling():
    """A user-chosen per-copy rotation is authoritative, never coupled."""
    from kicraft.layout_editor.model import ManualLayout, ManualLeafPlacement
    from kicraft.layout_editor.outline import OutlineSpec

    donor, replica = _driver_artifacts()
    manual = ManualLayout(
        placements=[
            ManualLeafPlacement(instance_path="/driver_a", origin=Point(0.0, 0.0), rotation=0.0),
            ManualLeafPlacement(instance_path="/driver_b", origin=Point(30.0, 0.0), rotation=90.0),
        ],
        outline=OutlineSpec.rect(Point(-5.0, -5.0), Point(60.0, 40.0)),
    )
    state, _ = _compose_artifacts(
        [donor, replica],
        spacing_mm=2.0,
        rotation_step_deg=0.0,
        parent_definition=None,
        pcb_path=None,
        cfg={},
        seed=0,
        manual_layout=manual,
    )
    assert state.packing_metadata["replica_rotation_groups"] == []
    rotations = {entry.instance_path: entry.rotation for entry in state.entries}
    assert rotations == {"/driver_a": 0.0, "/driver_b": 90.0}


def test_self_donor_is_an_error():
    self_replica = _artifact(
        "LOOP", _driver_leaf("U"), 20.0, 20.0, replicated_from="/loop"
    )
    with pytest.raises(ValueError, match=r"^replica_rotation_cycle:/loop$"):
        _replica_rotation_groups([self_replica], {0: "BLOCK_0"}, {})


def test_donor_cycle_is_an_error():
    first = _artifact("A", _driver_leaf("U"), 20.0, 20.0, replicated_from="/b")
    second = _artifact("B", _driver_leaf("R"), 20.0, 20.0, replicated_from="/a")
    with pytest.raises(ValueError, match=r"^replica_rotation_cycle:/"):
        _replica_rotation_groups([first, second], {0: "BLOCK_0", 1: "BLOCK_1"}, {})


# ---------------------------------------------------------------------------
# The hard validation seam


def _placements(donor, replica, donor_rotation: float, replica_rotation: float):
    return {
        donor.instance_path: ChildArtifactPlacement(
            artifact=donor, origin=Point(0.0, 0.0), rotation=donor_rotation
        ),
        replica.instance_path: ChildArtifactPlacement(
            artifact=replica, origin=Point(30.0, 0.0), rotation=replica_rotation
        ),
    }


def test_validate_replica_rotations_accepts_equal_angles():
    donor, replica = _driver_artifacts()
    _validate_replica_rotations(
        _placements(donor, replica, 180.0, 180.0), [["/driver_a", "/driver_b"]]
    )
    # 0 vs 360 must compare equal modulo 360.
    _validate_replica_rotations(
        _placements(donor, replica, 0.0, 360.0), [["/driver_a", "/driver_b"]]
    )


def test_validate_replica_rotations_rejects_mismatch_and_missing_member():
    donor, replica = _driver_artifacts()
    with pytest.raises(ValueError, match=r"^replica_rotation_mismatch:/driver_b$"):
        _validate_replica_rotations(
            _placements(donor, replica, 0.0, 90.0), [["/driver_a", "/driver_b"]]
        )
    with pytest.raises(
        ValueError, match=r"^replica_rotation_mismatch:/driver_b_missing$"
    ):
        _validate_replica_rotations(
            _placements(donor, replica, 0.0, 0.0), [["/driver_a", "/driver_b_missing"]]
        )


def test_injected_mismatched_recovery_hard_rejects_candidate(monkeypatch):
    """A mismatched recovered placement aborts compose, before stamping.

    The injected replica rotation is itself a valid cardinal angle with valid
    outline/courtyard geometry, so nothing but the replica invariant can
    reject it.
    """
    donor, replica = _driver_artifacts()
    neighbour = _wide_neighbour()
    artifacts = [donor, replica, neighbour]

    # Baseline: the same artifacts compose cleanly.
    baseline, _ = _compose_artifacts(
        artifacts,
        spacing_mm=2.0,
        rotation_step_deg=0.0,
        parent_definition=None,
        pcb_path=None,
        cfg={},
        seed=0,
    )
    assert baseline.packing_metadata["replica_rotation_groups"] == [
        ["/driver_a", "/driver_b"]
    ]

    real_recovery = compose_module.placements_from_solved_state

    def _mismatched(solved, loaded, synthetic_refs):
        placements = real_recovery(solved, loaded, synthetic_refs)
        placements["/driver_b"] = ChildArtifactPlacement(
            artifact=placements["/driver_b"].artifact,
            origin=placements["/driver_b"].origin,
            rotation=(placements["/driver_b"].rotation + 90.0) % 360.0,
        )
        return placements

    monkeypatch.setattr(
        compose_module, "placements_from_solved_state", _mismatched
    )
    with pytest.raises(ValueError, match=r"^replica_rotation_mismatch:/driver_b$"):
        _compose_artifacts(
            artifacts,
            spacing_mm=2.0,
            rotation_step_deg=0.0,
            parent_definition=None,
            pcb_path=None,
            cfg={},
            seed=0,
        )
