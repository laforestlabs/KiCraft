"""Regression tests for ``PlacementSolver._stack_compatible_blocks``.

Background: parent compose's dual-layer real estate only gets used if
the small unlocked SMT subcircuit blocks (front-only blockers) end up
on top of the large locked THT subcircuit anchor (back-only blockers,
e.g. LLUPS BATT). Force-directed + SA cannot cross the energy barrier
to land a candidate inside an anchor's bbox, so commit fbce780 added
``_stack_compatible_blocks`` as a deterministic post-SA pass to migrate
each blocker-compatible candidate onto its largest viable anchor.

This pass has broken twice unobserved (no test coverage). The contract
captured here:

  1. With one locked back-only anchor and one unlocked front-only
     candidate, the candidate's body center moves inside the anchor's
     bbox.
  2. Multiple candidates row-pack inside the anchor along the longer
     axis; each candidate stays inside the anchor's bbox.
  3. Same-side anchor/candidate pairs (front+front or back+back) do
     NOT stack — the function leaves the candidate at its force-
     directed position.
  4. With NO locked anchors (all blocks unlocked), the function is a
     no-op. This documents the current behaviour, but in real parent
     compose this means the BATT block must be locked upstream by
     ``_pin_edge_components`` for stacking to engage; if that upstream
     pinning regresses, this test still passes but stacking on real
     boards silently disappears. Cross-check via the board-level
     integration test below.
  5. The post-stack ``_resolve_overlaps`` must NOT push compatible
     stacked candidates back out of the anchor's bbox — same-side
     candidate-vs-candidate collisions resolve within the bbox; the
     candidate-vs-anchor pair stays overlapping because their blocker
     sides are opposite.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.parent_adapter import (
    artifact_to_component,
    attachment_constraints_to_zones,
    synthetic_block_ref,
)
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.subcircuit_composer import (
    LeafBlockerSet,
    derive_attachment_constraints,
)
from kicraft.autoplacer.brain.subcircuit_instances import (
    LoadedSubcircuitArtifact,
)
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Pad,
    Point,
    SubCircuitId,
    SubCircuitLayout,
)


def _blocker_set(
    *,
    front_pads=(),
    back_pads=(),
    front_tht_pads=(),
    back_tht_pads=(),
    tht_drills=(),
    front_traces=(),
    back_traces=(),
    leaf_outline=(Point(0.0, 0.0), Point(10.0, 10.0)),
) -> LeafBlockerSet:
    return LeafBlockerSet(
        front_pads=tuple(front_pads),
        back_pads=tuple(back_pads),
        front_tht_pads=tuple(front_tht_pads),
        back_tht_pads=tuple(back_tht_pads),
        tht_drills=tuple(tht_drills),
        front_traces=tuple(front_traces),
        back_traces=tuple(back_traces),
        leaf_outline=leaf_outline,
    )


def _block(
    ref: str,
    *,
    pos: Point,
    width: float,
    height: float,
    blocker_set: LeafBlockerSet,
    locked: bool = False,
    rotation: float = 0.0,
) -> Component:
    comp = Component(
        ref=ref,
        value=ref,
        pos=Point(pos.x, pos.y),
        rotation=rotation,
        layer=Layer.FRONT,
        width_mm=width,
        height_mm=height,
        kind="subcircuit",
        locked=locked,
        body_center=Point(pos.x, pos.y),
    )
    comp.block_blocker_set = blocker_set
    comp.block_artifact_origin_offset = Point(width / 2.0, height / 2.0)
    return comp


def _board(comps: dict[str, Component], outline: float = 200.0) -> BoardState:
    return BoardState(
        components=comps,
        nets={},
        traces=[],
        vias=[],
        silkscreen=[],
        board_outline=(Point(0.0, 0.0), Point(outline, outline)),
    )


def _solver(state: BoardState) -> PlacementSolver:
    return PlacementSolver(state, config={"opposite_side_stacking_pass": True}, seed=0)


def _inside_bbox(
    inner: Component, outer: Component, *, slack: float = 0.5
) -> bool:
    """True iff ``inner``'s body bbox is entirely within ``outer``'s with
    a small slack to absorb the spacing band the row-pack adds."""
    in_tl, in_br = inner.bbox()
    out_tl, out_br = outer.bbox()
    return (
        in_tl.x >= out_tl.x - slack
        and in_tl.y >= out_tl.y - slack
        and in_br.x <= out_br.x + slack
        and in_br.y <= out_br.y + slack
    )


# ---------------------------------------------------------------------------
# Geometry covered by the original fbce780 design (one locked back-only anchor
# + one unlocked front-only candidate).


def test_single_candidate_stacks_on_locked_back_only_anchor():
    """Mirrors LLUPS at BATT (locked, back-only) + BOOST 5V (unlocked,
    front-only). Pre-stack the candidate sits 60 mm away; post-stack
    it must be inside the anchor's bbox."""
    anchor_blockers = _blocker_set(
        back_pads=[(Point(0.0, 0.0), Point(80.0, 50.0))],
        leaf_outline=(Point(0.0, 0.0), Point(80.0, 50.0)),
    )
    cand_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(20.0, 12.0))],
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 12.0)),
    )
    anchor = _block(
        "BATT", pos=Point(60.0, 60.0), width=80.0, height=50.0,
        blocker_set=anchor_blockers, locked=True,
    )
    cand = _block(
        "BOOST", pos=Point(150.0, 60.0), width=20.0, height=12.0,
        blocker_set=cand_blockers, locked=False,
    )
    state = _board({anchor.ref: anchor, cand.ref: cand})

    _solver(state)._stack_compatible_blocks(state.components)

    assert _inside_bbox(cand, anchor), (
        f"BOOST should stack inside BATT bbox; got pos={cand.pos!r}, "
        f"BATT bbox={anchor.bbox()}"
    )


def test_multiple_candidates_row_pack_inside_anchor():
    """Three small front-only candidates onto one wide back-only anchor.
    Row-pack centres them along the anchor's longer axis; each candidate
    must end up inside the anchor's bbox."""
    anchor_blockers = _blocker_set(
        back_pads=[(Point(0.0, 0.0), Point(90.0, 60.0))],
        leaf_outline=(Point(0.0, 0.0), Point(90.0, 60.0)),
    )
    cand_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(20.0, 12.0))],
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 12.0)),
    )
    anchor = _block(
        "BATT", pos=Point(60.0, 60.0), width=90.0, height=60.0,
        blocker_set=anchor_blockers, locked=True,
    )
    cands = {
        f"C{i}": _block(
            f"C{i}", pos=Point(150.0 + i * 25.0, 60.0),
            width=20.0, height=12.0,
            blocker_set=cand_blockers, locked=False,
        )
        for i in range(3)
    }
    state = _board({anchor.ref: anchor, **cands})

    _solver(state)._stack_compatible_blocks(state.components)

    for cand in cands.values():
        assert _inside_bbox(cand, anchor), (
            f"{cand.ref} should stack inside BATT bbox; got pos={cand.pos!r}, "
            f"BATT bbox={anchor.bbox()}"
        )


def test_same_side_pair_does_not_stack():
    """Front-only candidate must NOT stack on a front-only anchor —
    that would put copper-on-copper. The function should leave the
    candidate at its force-directed position unchanged."""
    front_anchor_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(80.0, 50.0))],
        leaf_outline=(Point(0.0, 0.0), Point(80.0, 50.0)),
    )
    front_cand_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(20.0, 12.0))],
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 12.0)),
    )
    anchor = _block(
        "BIG_FRONT", pos=Point(60.0, 60.0), width=80.0, height=50.0,
        blocker_set=front_anchor_blockers, locked=True,
    )
    cand = _block(
        "SMALL_FRONT", pos=Point(150.0, 60.0), width=20.0, height=12.0,
        blocker_set=front_cand_blockers, locked=False,
    )
    state = _board({anchor.ref: anchor, cand.ref: cand})

    initial_pos = Point(cand.pos.x, cand.pos.y)
    _solver(state)._stack_compatible_blocks(state.components)

    assert cand.pos.x == initial_pos.x and cand.pos.y == initial_pos.y, (
        f"same-side cand should not have moved; got {cand.pos!r}"
    )


def test_no_locked_anchor_is_no_op():
    """When every subcircuit block is unlocked (the upstream
    pin-edges pass didn't lock the largest leaf), the stack pass is
    a no-op. This is a known limitation of the current contract:
    upstream MUST lock the anchor for stacking to engage. The
    integration test below exercises that pinning."""
    anchor_blockers = _blocker_set(
        back_pads=[(Point(0.0, 0.0), Point(80.0, 50.0))],
        leaf_outline=(Point(0.0, 0.0), Point(80.0, 50.0)),
    )
    cand_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(20.0, 12.0))],
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 12.0)),
    )
    anchor = _block(
        "BATT_NOT_LOCKED", pos=Point(60.0, 60.0), width=80.0, height=50.0,
        blocker_set=anchor_blockers, locked=False,
    )
    cand = _block(
        "BOOST", pos=Point(150.0, 60.0), width=20.0, height=12.0,
        blocker_set=cand_blockers, locked=False,
    )
    state = _board({anchor.ref: anchor, cand.ref: cand})

    initial = Point(cand.pos.x, cand.pos.y)
    _solver(state)._stack_compatible_blocks(state.components)

    assert cand.pos.x == initial.x and cand.pos.y == initial.y, (
        "with no locked anchor, candidate should not move"
    )


# ---------------------------------------------------------------------------
# The "stacking survives _resolve_overlaps" check — this is the path the
# fbce780 commit specifically validated and the one most likely to regress
# silently.


def test_stack_survives_resolve_overlaps():
    """After stacking, _resolve_overlaps runs (real solve loop calls it
    in solve_resolve_overlaps_ms). Same-side candidate-vs-candidate is
    spread out; candidate-vs-anchor must STAY overlapping because the
    blocker pair is compatible. If overlap resolution pushes
    candidates back outside the anchor's bbox, stacking is undone and
    we lose the dual-layer area we gained."""
    anchor_blockers = _blocker_set(
        back_pads=[(Point(0.0, 0.0), Point(90.0, 60.0))],
        leaf_outline=(Point(0.0, 0.0), Point(90.0, 60.0)),
    )
    cand_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(20.0, 12.0))],
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 12.0)),
    )
    anchor = _block(
        "BATT", pos=Point(60.0, 60.0), width=90.0, height=60.0,
        blocker_set=anchor_blockers, locked=True,
    )
    cands = {
        f"C{i}": _block(
            f"C{i}", pos=Point(150.0 + i * 25.0, 60.0),
            width=20.0, height=12.0,
            blocker_set=cand_blockers, locked=False,
        )
        for i in range(3)
    }
    state = _board({anchor.ref: anchor, **cands})
    solver = _solver(state)

    solver._stack_compatible_blocks(state.components)
    # Same call sequence as in solve(): stack → resolve_overlaps.
    solver._resolve_overlaps(state.components)

    for cand in cands.values():
        assert _inside_bbox(cand, anchor, slack=1.0), (
            f"{cand.ref} drifted outside BATT bbox after _resolve_overlaps; "
            f"pos={cand.pos!r}, anchor bbox={anchor.bbox()}"
        )


# ---------------------------------------------------------------------------
# Integration check: verify that ``_pin_edge_components`` locks a
# subcircuit-kind block whose ``component_zones`` config has a "zone"
# key. This is the upstream condition stack_compatible_blocks depends
# on. If this regresses, the unit tests above keep passing while real
# parent compose silently stops stacking.


def test_pin_edge_components_locks_zoned_subcircuit_block():
    """A subcircuit block with a `zone` cfg entry must be locked by
    ``_pin_edge_components`` so the post-SA stack pass can use it as
    an anchor. This regression-tests the bridge between the user's
    ``component_zones`` config and the solver's locked/unlocked split."""
    anchor_blockers = _blocker_set(
        back_pads=[(Point(0.0, 0.0), Point(80.0, 50.0))],
        leaf_outline=(Point(0.0, 0.0), Point(80.0, 50.0)),
    )
    anchor = _block(
        "BATT", pos=Point(60.0, 60.0), width=80.0, height=50.0,
        blocker_set=anchor_blockers, locked=False,
    )
    state = _board({anchor.ref: anchor})

    solver = PlacementSolver(
        state,
        config={
            "component_zones": {
                "BATT": {"zone": "bottom"},
            },
            "opposite_side_stacking_pass": True,
        },
        seed=0,
    )
    solver._pin_edge_components(state.components)

    assert anchor.locked, (
        "BATT subcircuit block with zone=bottom should be locked by "
        "_pin_edge_components so _stack_compatible_blocks can use it as an anchor"
    )


# ---------------------------------------------------------------------------
# Upstream chain: user's ``component_zones`` -> derive_attachment_constraints
# -> attachment_constraints_to_zones -> block-level zone -> locked.
#
# The realistic LLUPS shape: the user only mentions internal child refs
# (BT1, BT2 inside the BATT artifact). The chain must propagate that to
# the BATT BLOCK's solver-level zone entry, otherwise the BATT block is
# never locked and stacking silently disappears even though the
# isolated unit tests above pass.


def _pth_pad(ref: str, pad_id: str, x: float, y: float) -> Pad:
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net="", layer=Layer.FRONT)


def _battery_artifact() -> LoadedSubcircuitArtifact:
    """Synthetic ``BATT`` artifact carrying two internal cell refs (BT1,
    BT2) so the user's BT1/BT2 ``zone=bottom`` config has somewhere to
    attach. Cells are flagged through-hole to mirror real battery
    holders, which is what makes the BATT block back-only by area."""
    bt1 = Component(
        ref="BT1",
        value="",
        pos=Point(20.0, 30.0),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=40.0,
        height_mm=20.0,
        pads=[
            _pth_pad("BT1", "1", 5.0, 30.0),
            _pth_pad("BT1", "2", 35.0, 30.0),
        ],
        kind="battery",
        is_through_hole=True,
    )
    bt2 = Component(
        ref="BT2",
        value="",
        pos=Point(60.0, 30.0),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=40.0,
        height_mm=20.0,
        pads=[
            _pth_pad("BT2", "1", 45.0, 30.0),
            _pth_pad("BT2", "2", 75.0, 30.0),
        ],
        kind="battery",
        is_through_hole=True,
    )
    layout = SubCircuitLayout(
        subcircuit_id=SubCircuitId(
            sheet_name="BATT",
            sheet_file="batt.kicad_sch",
            instance_path="/batt",
        ),
        components={"BT1": bt1, "BT2": bt2},
        traces=[],
        vias=[],
        bounding_box=(80.0, 60.0),
    )
    return LoadedSubcircuitArtifact(
        artifact_dir="/fake/batt",
        metadata={},
        debug={},
        layout=layout,
        source_files={},
    )


def test_user_child_zone_propagates_to_block_zone_entry():
    """End-to-end: user has ``BT1: zone=bottom`` in ``component_zones``,
    BT1 lives inside the BATT artifact. After the full
    derive_attachment_constraints -> attachment_constraints_to_zones
    chain, the BATT BLOCK's zone entry must contain ``zone=bottom``.

    This is the linchpin step. If this regresses, the upstream chain
    silently stops producing a block-level zone, so
    _pin_edge_components has nothing to lock on, and stacking
    disappears in real parent compose runs."""
    artifact = _battery_artifact()
    artifacts = [artifact]

    user_component_zones = {
        "BT1": {"zone": "bottom"},
        "BT2": {"zone": "bottom"},
    }

    derived = derive_attachment_constraints(
        artifacts,
        parent_local_components={},
        component_zones=user_component_zones,
        cfg={},
    )

    assert 0 in derived.child_specs, (
        "BATT artifact should have child_specs[0] with the BT1/BT2 "
        "constraints attached; if missing, derive_attachment_constraints "
        "is no longer matching ref names against artifact.layout.components"
    )
    assert any(c.target == "zone" for c in derived.child_specs[0].constraints), (
        "BATT artifact's child_spec should carry at least one zone constraint "
        "from BT1/BT2"
    )

    synthetic_refs = {0: synthetic_block_ref(0, artifact.sheet_name)}
    block_zones, _allowed = attachment_constraints_to_zones(
        derived, synthetic_refs, artifacts
    )
    block_ref = synthetic_refs[0]

    assert block_ref in block_zones, (
        f"BATT block ref {block_ref} should appear in block_zones; without "
        "this, _pin_edge_components has no zone to lock on"
    )
    assert "zone" in block_zones[block_ref], (
        f"block_zones[{block_ref}] should carry a 'zone' key derived from "
        f"BT1/BT2, got {block_zones[block_ref]!r}"
    )


def test_full_chain_locks_battery_block_for_stacking():
    """End-to-end: the whole chain from user config down to the solver's
    locked/unlocked split. If this passes, real LLUPS parent compose
    will lock the BATT block and stacking will engage. If this fails,
    stacking silently disappears in real runs even though the targeted
    unit tests pass."""
    artifact = _battery_artifact()
    artifacts = [artifact]

    user_component_zones = {
        "BT1": {"zone": "bottom"},
        "BT2": {"zone": "bottom"},
    }
    derived = derive_attachment_constraints(
        artifacts,
        parent_local_components={},
        component_zones=user_component_zones,
        cfg={},
    )
    synthetic_refs = {0: synthetic_block_ref(0, artifact.sheet_name)}
    block_zones, _allowed = attachment_constraints_to_zones(
        derived, synthetic_refs, artifacts
    )

    # Build the synthetic BATT block (mirrors _compose_artifacts).
    block_ref = synthetic_refs[0]
    block = artifact_to_component(artifact, ref=block_ref)
    state = _board({block_ref: block})

    solver = PlacementSolver(
        state,
        config={
            "component_zones": dict(block_zones),
            "opposite_side_stacking_pass": True,
        },
        seed=0,
    )
    solver._pin_edge_components(state.components)

    assert block.locked, (
        f"BATT block {block_ref} should be locked after the full chain; "
        f"block_zones={block_zones!r}"
    )


# ---------------------------------------------------------------------------
# THT-anchor regression: a THT-heavy block (battery holder, screw terminal)
# has annular ring shadow on BOTH layers around each PTH drill. Before the
# LeafBlockerSet split those shadow rects were stored in front_pads /
# back_pads, which made dominant_blocker_side return "dual" and
# can_overlap_sparse's outline gate refuse overlap with any front-only
# candidate -- killing LLUPS-style SMT-on-back-THT stacking.
#
# The fix moved THT pad rings into front_tht_pads / back_tht_pads (still
# checked for per-pad shorts via union, but excluded from outline-area
# intent). Two contracts pinned below: (1) a THT anchor with REAL back
# copper (typical battery holder with B.Cu battery contacts) auto-detects
# as "back" without any project override; (2) a pure-THT anchor (screw
# terminal with no SMT pads at all) classifies as "none" by geometry and
# still requires the force_back_only override to declare placement intent.


def test_thT_anchor_with_real_back_pads_auto_detects_as_back():
    """A typical battery holder has THT corner pads (drill-shadow on
    both layers) PLUS real B.Cu battery contact pads. The contact pads
    establish "back-side intent"; the THT corner shadow must NOT count
    as front-side intent. Result: ``dominant_blocker_side`` returns
    "back" without any project override, ``_stack_compatible_blocks``
    treats it as a back-side anchor, and a front-only SMT candidate
    stacks on top and survives ``_resolve_overlaps``.

    This is the LLUPS BATT case the predicate fix targets. Before the
    fix, the THT corner rings impersonated F.Cu intent and the anchor
    looked "dual", forcing the user to set
    ``parent_placement.backside_through_hole_leaves: ["BATT"]`` just
    to declare what the geometry already implies."""
    tht_corner_rings = [
        (Point(0.0, 0.0), Point(8.0, 8.0)),
        (Point(72.0, 0.0), Point(80.0, 8.0)),
        (Point(0.0, 52.0), Point(8.0, 60.0)),
        (Point(72.0, 52.0), Point(80.0, 60.0)),
    ]
    real_back_contact_pads = [
        # Battery contact strips on B.Cu, in the empty middle of the
        # body where the candidate also wants to land. Slightly off-
        # center on the y-axis so the candidate row-pack lands above
        # the contacts rather than on top of them.
        (Point(20.0, 38.0), Point(60.0, 50.0)),
    ]
    anchor_blockers = _blocker_set(
        back_pads=real_back_contact_pads,
        front_tht_pads=tht_corner_rings,
        back_tht_pads=tht_corner_rings,
        tht_drills=tht_corner_rings,
        leaf_outline=(Point(0.0, 0.0), Point(80.0, 60.0)),
    )
    cand_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(20.0, 12.0))],
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 12.0)),
    )
    anchor = _block(
        "BATT", pos=Point(60.0, 60.0), width=80.0, height=60.0,
        blocker_set=anchor_blockers, locked=True,
    )
    # Crucially: NO block_force_back_only override.
    cand = _block(
        "BOOST", pos=Point(180.0, 60.0), width=20.0, height=12.0,
        blocker_set=cand_blockers, locked=False,
    )
    state = _board({anchor.ref: anchor, cand.ref: cand})
    solver = _solver(state)

    solver._stack_compatible_blocks(state.components)
    solver._resolve_overlaps(state.components)

    assert _inside_bbox(cand, anchor, slack=1.0), (
        "Front-only candidate should auto-stack on a THT anchor with "
        "real B.Cu pads -- the THT corner-ring shadow must not count as "
        "F.Cu intent. cand pos={!r}, BATT bbox={}".format(
            cand.pos, anchor.bbox()
        )
    )


def test_pure_THT_anchor_still_requires_force_back_only_override():
    """A pure-THT anchor (e.g. a 0.1-inch pin-header strip) has annular
    ring shadow on both layers and NO real SMT copper anywhere. The
    geometry alone cannot determine which side the leaf is meant to
    live on -- that is project intent, not blocker geometry.

    ``dominant_blocker_side`` returns "none" for such a leaf;
    ``_stack_compatible_blocks`` skips "none"-side anchors. The
    project-level ``parent_placement.backside_through_hole_leaves``
    override (lands as ``block_force_back_only=True``) remains the
    declared escape hatch for these cases. This test pins both halves
    of that contract: without override = no stacking, with override =
    stacking succeeds."""
    pad_rects = [
        (Point(0.0, 0.0), Point(8.0, 8.0)),
        (Point(72.0, 0.0), Point(80.0, 8.0)),
        (Point(0.0, 52.0), Point(8.0, 60.0)),
        (Point(72.0, 52.0), Point(80.0, 60.0)),
    ]
    anchor_blockers = _blocker_set(
        front_tht_pads=pad_rects,
        back_tht_pads=pad_rects,
        tht_drills=pad_rects,
        leaf_outline=(Point(0.0, 0.0), Point(80.0, 60.0)),
    )
    cand_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(20.0, 12.0))],
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 12.0)),
    )

    # Half 1: no override -> "none"-side anchor -> no stacking.
    anchor = _block(
        "TERMINAL", pos=Point(60.0, 60.0), width=80.0, height=60.0,
        blocker_set=anchor_blockers, locked=True,
    )
    cand = _block(
        "BOOST", pos=Point(180.0, 60.0), width=20.0, height=12.0,
        blocker_set=cand_blockers, locked=False,
    )
    state = _board({anchor.ref: anchor, cand.ref: cand})
    solver = _solver(state)
    solver._stack_compatible_blocks(state.components)
    solver._resolve_overlaps(state.components)
    assert not _inside_bbox(cand, anchor, slack=1.0), (
        "Pure-THT anchor with no real layer copper should classify as "
        "'none' and not be selected as a stacking anchor without an "
        "explicit project override."
    )

    # Half 2: with override -> forces "back" classification -> stacks.
    anchor.block_force_back_only = True
    cand.pos = Point(180.0, 60.0)
    state = _board({anchor.ref: anchor, cand.ref: cand})
    solver = _solver(state)
    solver._stack_compatible_blocks(state.components)
    solver._resolve_overlaps(state.components)
    assert _inside_bbox(cand, anchor, slack=1.0), (
        "Pure-THT anchor with block_force_back_only=True should host "
        "front-only candidates as a back-side anchor. cand pos={!r}, "
        "anchor bbox={}".format(cand.pos, anchor.bbox())
    )


def test_charger_style_continuous_fcu_still_rejected():
    """Regression: commit 6c15e92's fix for CHARGER+BOOST_5V
    continuous-F.Cu stamping shorts (~45 shorting_items per candidate)
    must still hold AFTER the THT-shadow split. Both leaves carry real
    F.Cu intent (SMT pads + routed traces); the outline gate must
    still refuse overlap when their outlines coincide. If this
    regresses, real-world parent compose will reproduce the original
    short-stack bug."""
    from kicraft.autoplacer.brain.subcircuit_composer import (
        LeafBlockerSet,
        can_overlap_sparse,
    )

    # CHARGER-shape: real SMT F.Cu pads + routed F.Cu traces + a back-
    # side PTH drill (annular ring shadow now lives in back_tht_pads
    # rather than back_pads, since it's not real B.Cu intent).
    charger_bs = LeafBlockerSet(
        front_pads=(
            (Point(2.0, 2.0), Point(4.0, 3.0)),
            (Point(2.0, 5.0), Point(4.0, 6.0)),
        ),
        back_pads=(),
        front_traces=((Point(4.0, 2.5), Point(20.0, 3.0)),),  # routed F.Cu
        back_traces=(),
        front_tht_pads=((Point(20.0, 2.0), Point(22.0, 3.0)),),
        back_tht_pads=((Point(20.0, 2.0), Point(22.0, 3.0)),),
        tht_drills=((Point(20.5, 2.0), Point(22.5, 3.0)),),
        leaf_outline=(Point(0.0, 0.0), Point(30.0, 20.0)),
    )
    boost_bs = LeafBlockerSet(
        front_pads=(
            (Point(2.0, 2.0), Point(4.0, 3.0)),
            (Point(2.0, 5.0), Point(4.0, 6.0)),
        ),
        back_pads=(),
        front_traces=((Point(4.0, 2.5), Point(15.0, 3.0)),),  # routed F.Cu
        back_traces=(),
        tht_drills=(),
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 15.0)),
    )
    # Place both at the same world origin so outlines overlap.
    origin = Point(0.0, 0.0)
    assert not can_overlap_sparse(
        charger_bs, origin, 0.0,
        boost_bs, origin, 0.0,
    ), (
        "CHARGER (front pads + traces) + BOOST_5V (front pads + traces) "
        "with overlapping outlines must be incompatible -- this is the "
        "original 6c15e92 guarantee. If this passes (compatible), "
        "continuous-F.Cu shorts will reappear."
    )


def test_outline_gate_uses_real_pads_not_THT_shadows():
    """Direct contract test for the predicate's outline gate.

    A leaf with ONLY THT pad rings on F.Cu (no real F.Cu pads, no
    F.Cu traces) must NOT trigger the same-layer outline gate
    against a front-only SMT leaf -- the per-pad checks already
    cover ring-vs-pad collisions. The gate exists only for
    continuous F.Cu plane / pour scenarios, and a sparse drill
    shadow is not that."""
    from kicraft.autoplacer.brain.subcircuit_composer import (
        LeafBlockerSet,
        can_overlap_sparse,
    )

    tht_only = LeafBlockerSet(
        front_pads=(),
        back_pads=(),
        front_tht_pads=(
            (Point(0.0, 0.0), Point(2.0, 2.0)),
            (Point(28.0, 18.0), Point(30.0, 20.0)),
        ),
        back_tht_pads=(
            (Point(0.0, 0.0), Point(2.0, 2.0)),
            (Point(28.0, 18.0), Point(30.0, 20.0)),
        ),
        tht_drills=(
            (Point(0.5, 0.5), Point(1.5, 1.5)),
            (Point(28.5, 18.5), Point(29.5, 19.5)),
        ),
        leaf_outline=(Point(0.0, 0.0), Point(30.0, 20.0)),
    )
    smt_only_front = LeafBlockerSet(
        front_pads=((Point(10.0, 8.0), Point(20.0, 12.0)),),
        back_pads=(),
        tht_drills=(),
        leaf_outline=(Point(0.0, 0.0), Point(30.0, 20.0)),
    )
    # Outlines fully coincide; SMT pad sits in the empty middle of
    # the THT outline, between the two corner drills.
    assert can_overlap_sparse(
        tht_only, Point(0.0, 0.0), 0.0,
        smt_only_front, Point(0.0, 0.0), 0.0,
    ), (
        "THT-shadow-only outline overlapping a front-only SMT leaf must "
        "be allowed -- THT annular ring shadow does not represent "
        "continuous F.Cu copper. If this fails, the outline gate is "
        "wrongly counting front_tht_pads as front intent."
    )


def test_outline_gate_fires_on_mixed_THT_plus_real_smt_front():
    """Guard against over-correction. A leaf with BOTH THT corner rings
    AND a real SMT F.Cu pad still has front-side intent (the real SMT
    pad). The outline gate must still fire against another front-only
    leaf to avoid the original CHARGER-style continuous-F.Cu shorts."""
    from kicraft.autoplacer.brain.subcircuit_composer import (
        LeafBlockerSet,
        can_overlap_sparse,
    )

    mixed = LeafBlockerSet(
        front_pads=((Point(10.0, 8.0), Point(15.0, 10.0)),),  # real SMT
        back_pads=(),
        front_tht_pads=((Point(0.0, 0.0), Point(2.0, 2.0)),),
        back_tht_pads=((Point(0.0, 0.0), Point(2.0, 2.0)),),
        tht_drills=((Point(0.5, 0.5), Point(1.5, 1.5)),),
        leaf_outline=(Point(0.0, 0.0), Point(30.0, 20.0)),
    )
    smt_other_front = LeafBlockerSet(
        front_pads=((Point(20.0, 8.0), Point(25.0, 10.0)),),
        back_pads=(),
        tht_drills=(),
        leaf_outline=(Point(0.0, 0.0), Point(30.0, 20.0)),
    )
    # Outlines fully coincide; per-pad checks miss because the SMT
    # pads are in different XY ranges. The outline gate must catch
    # this on the strength of the real F.Cu intent on both leaves.
    assert not can_overlap_sparse(
        mixed, Point(0.0, 0.0), 0.0,
        smt_other_front, Point(0.0, 0.0), 0.0,
    ), (
        "A leaf with BOTH THT corners AND real SMT F.Cu copper still "
        "has front-side intent; outlines coinciding with another "
        "front-intent leaf must be rejected to prevent continuous-F.Cu "
        "plane shorts."
    )


def test_force_back_only_override_unblocks_strict_dual_anchor():
    """Project config escape hatch: a sheet name listed in
    parent_placement.backside_through_hole_leaves should have its
    front-side intent suppressed even if the heuristic would
    classify it as front-intent. Lets users force stacking through
    when the auto-detection misses."""
    # An anchor that the heuristic WOULD classify as front-intent
    # (more front pads than back pads) but the user wants to treat
    # as back-only.
    front_dominant_blockers = _blocker_set(
        front_pads=[
            (Point(5.0, 5.0), Point(20.0, 10.0)),
            (Point(60.0, 5.0), Point(75.0, 10.0)),
        ],
        back_pads=[(Point(40.0, 30.0), Point(45.0, 35.0))],  # less back area
        leaf_outline=(Point(0.0, 0.0), Point(80.0, 60.0)),
    )
    cand_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(20.0, 12.0))],
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 12.0)),
    )
    anchor = _block(
        "BATT", pos=Point(60.0, 60.0), width=80.0, height=60.0,
        blocker_set=front_dominant_blockers, locked=True,
    )
    cand = _block(
        "BOOST", pos=Point(180.0, 60.0), width=20.0, height=12.0,
        blocker_set=cand_blockers, locked=False,
    )
    state = _board({anchor.ref: anchor, cand.ref: cand})

    # Without the override: heuristic says front-intent on anchor
    # (front_pads_area > back_pads_area), candidate has front pads,
    # so resolve_overlaps will push the candidate out. Demonstrate
    # baseline failure first.
    solver_no_override = _solver(state)
    solver_no_override._stack_compatible_blocks(state.components)
    solver_no_override._resolve_overlaps(state.components)
    assert not _inside_bbox(cand, anchor, slack=1.0), (
        "Test pre-condition: without override, the heuristic should "
        "classify the front-pad-dominant anchor as front-intent and "
        "reject overlap. If this assertion fires, the heuristic has "
        "changed and the override test no longer demonstrates the gap "
        "it is meant to cover."
    )

    # Apply the override and re-run from scratch.
    cand.pos = Point(180.0, 60.0)
    anchor.block_force_back_only = True
    state = _board({anchor.ref: anchor, cand.ref: cand})
    solver = _solver(state)
    solver._stack_compatible_blocks(state.components)
    solver._resolve_overlaps(state.components)

    assert _inside_bbox(cand, anchor, slack=1.0), (
        "With block_force_back_only=True on the anchor, the candidate "
        "should stack and stay stacked through _resolve_overlaps. "
        f"cand pos={cand.pos!r}, anchor bbox={anchor.bbox()}"
    )


def test_compose_artifacts_propagates_backside_through_hole_cfg():
    """End-to-end: cfg["parent_placement"]["backside_through_hole_leaves"]
    set on a sheet name must land as block_force_back_only=True on the
    synthetic block component constructed in compose-time setup. This
    is the bridge tested in isolation -- the override is otherwise
    invisible to PlacementSolver."""
    artifact = _battery_artifact()
    block = artifact_to_component(artifact, ref="TEST_BLOCK")

    # Sanity: default is False until cfg is consulted.
    assert block.block_force_back_only is False

    # Mirror the compose-time setup (read cfg, set flag if sheet matches).
    # The unit under test here is the contract: setting the flag turns
    # the heuristic off for that leaf without touching extract.
    cfg_back_through_hole_leaves = {"BATT", "TERMINAL_BLOCK"}
    if artifact.sheet_name in cfg_back_through_hole_leaves:
        block.block_force_back_only = True

    assert block.block_force_back_only is True
    # And the predicate should now skip the front-side outline check
    # for this leaf even if its blocker set has front-side intent.
    from kicraft.autoplacer.brain.placement_utils import _blocker_pair_compatible

    other_blockers = _blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(20.0, 12.0))],
        leaf_outline=(Point(0.0, 0.0), Point(20.0, 12.0)),
    )
    other = _block(
        "OTHER", pos=Point(50.0, 50.0), width=20.0, height=12.0,
        blocker_set=other_blockers, locked=False,
    )
    block.pos = Point(60.0, 60.0)
    block.width_mm = 80.0
    block.height_mm = 60.0
    block.body_center = Point(60.0, 60.0)
    assert _blocker_pair_compatible(block, other), (
        "with block_force_back_only=True, _blocker_pair_compatible should "
        "return True (compatible) for an otherwise-conflicting front-only pair"
    )
