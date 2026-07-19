"""USB-C / edge-connector placement: face outward, sit at the edge, overhang.

Covers the systematic fix for edge connectors ending up unusable:
  * Layer A -- detect_opening_direction reads the connector mouth from real
    footprint geometry.
  * Layer 0 -- shared edge_outward_angle / opening_board_angle helpers.
  * Layer B -- the composer narrows a leaf's rotation candidates so an
    embedded connector's mouth faces outward at its assigned edge.
  * Layer C -- the board edge sits flush-to-overhanging the mouth, never
    inset past it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.subcircuit_composer import (
    AttachmentConstraint,
    _filter_rotations_for_connector_opening,
    edge_anchor_target_coordinate,
)
from kicraft.autoplacer.brain.types import (
    Component,
    Layer,
    Point,
    angles_close,
    edge_outward_angle,
    opening_board_angle,
)

import logging

_LOGGER = logging.getLogger(__name__)


# --- Layer 0: shared helpers ---------------------------------------------


def test_edge_outward_angle_front_and_back():
    # Front: opening should point away from the board interior.
    assert edge_outward_angle(Layer.FRONT, "bottom") == 90.0
    assert edge_outward_angle(Layer.FRONT, "top") == 270.0
    assert edge_outward_angle(Layer.FRONT, "left") == 180.0
    assert edge_outward_angle(Layer.FRONT, "right") == 0.0
    # Back: Flip() mirrors local X, so left/right swap; top/bottom unchanged.
    assert edge_outward_angle(Layer.BACK, "left") == 0.0
    assert edge_outward_angle(Layer.BACK, "right") == 180.0
    assert edge_outward_angle(Layer.BACK, "bottom") == 90.0


def test_opening_board_angle_inverts_rotation():
    # local = (board + rotation)  =>  board = (local - rotation)
    assert opening_board_angle(90.0, 0.0) == 90.0
    assert opening_board_angle(90.0, 90.0) == 0.0
    assert opening_board_angle(90.0, 270.0) == 180.0


def test_angles_close_wraps():
    assert angles_close(359.5, 0.2)
    assert angles_close(90.0, 90.4)
    assert not angles_close(90.0, 180.0)


# --- Layer A: real footprint geometry ------------------------------------


def test_detect_opening_direction_real_bnc_elbow():
    """KC-MUSEUD regression: the BNC elbow jack's BODY extends +y over the
    board while its mating barrel points -y past the pads -- the opposite of
    the USB-C shell-overhang pattern below, so the body-overhang heuristic
    read its mouth 180 deg wrong and the placer pointed the barrel INBOARD
    on every board using the part. The footprint now carries the
    authoritative 'Board Edge' Dwgs.User marker (detection rule 1); this
    pins the detected direction to the mating side."""
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction

    lib = "kicraft/parts_library/bnc-pcb-jack/bnc-pcb-jack.pretty"
    fp = pcbnew.FootprintLoad(lib, "ANT-TH_KH-BNC50-3511")
    assert fp is not None
    assert detect_opening_direction(fp) == 270.0
    # Local direction is invariant to the footprint's board orientation.
    for rot in (90.0, 180.0, 270.0):
        fp.SetOrientationDegrees(rot)
        assert detect_opening_direction(fp) == 270.0


def test_detect_opening_direction_real_usb_c():
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction

    lib = "kicraft/parts_library/usb-c-16p/usb-c-16p.pretty"
    fp = pcbnew.FootprintLoad(lib, "USB-C_SMD-TYPE-C-31-M-12_1")
    assert fp is not None
    # Pins/tail at -y, shell extends to +y: the mouth faces +y (90 deg local).
    assert detect_opening_direction(fp) == 90.0
    # opening_direction is footprint-local => invariant to the footprint's own
    # board orientation.
    for rot in (90.0, 180.0, 270.0):
        fp.SetOrientationDegrees(rot)
        assert detect_opening_direction(fp) == 90.0


# --- Single-board solver rotation ----------------------------------------


def _connector(opening: float | None, layer: Layer = Layer.FRONT) -> Component:
    return Component(
        ref="J1",
        value="USB-C",
        pos=Point(0.0, 0.0),
        rotation=0.0,
        layer=layer,
        width_mm=9.0,
        height_mm=3.0,
        kind="connector",
        pads=[],
        opening_direction=opening,
    )


@pytest.mark.parametrize(
    "edge,expected_rot",
    [("bottom", 0.0), ("right", 90.0), ("top", 180.0), ("left", 270.0)],
)
def test_best_rotation_points_mouth_outward(edge, expected_rot):
    comp = _connector(opening=90.0)
    rot = PlacementSolver._best_rotation_for_edge(comp, edge)
    assert rot == expected_rot
    # Sanity: rotating the body by `rot` puts the mouth at the outward angle.
    assert angles_close(
        opening_board_angle(comp.opening_direction, rot),
        edge_outward_angle(Layer.FRONT, edge),
    )


# --- Layer B: composer rotation filter -----------------------------------


def _spec_with_connector(opening, edge, layer=Layer.FRONT, leaf_local_rot=0.0):
    """A PlacementSpec stand-in: one leaf, one connector, four rotations.

    Only the attributes the filter touches are populated. Each candidate leaf
    rotation R yields a transformed connector at parent rotation
    (leaf_local_rot + R).
    """
    constraint = AttachmentConstraint(
        ref="J1",
        target="edge",
        value=edge,
        inward_keep_in_mm=0.0,
        outward_overhang_mm=0.5,
        source="child_artifact",
        child_index=0,
    )
    candidates = [0.0, 90.0, 180.0, 270.0]
    models = {
        rot: SimpleNamespace(
            transformed=SimpleNamespace(
                transformed_components={
                    "J1": Component(
                        ref="J1",
                        value="USB-C",
                        pos=Point(0.0, 0.0),
                        rotation=(leaf_local_rot + rot) % 360.0,
                        layer=layer,
                        width_mm=9.0,
                        height_mm=3.0,
                        kind="connector",
                        pads=[],
                        opening_direction=opening,
                    )
                }
            )
        )
        for rot in candidates
    }
    return SimpleNamespace(
        constraints=[constraint],
        rotation_candidates=list(candidates),
        all_rotation_candidates=list(candidates),
        models=models,
        instance_path="/leaf",
    )


@pytest.mark.parametrize(
    "edge,kept",
    [("bottom", 0.0), ("right", 90.0), ("top", 180.0), ("left", 270.0)],
)
def test_filter_keeps_only_outward_rotation(edge, kept):
    spec = _spec_with_connector(opening=90.0, edge=edge)
    _filter_rotations_for_connector_opening(spec, _LOGGER)
    assert spec.rotation_candidates == [kept]
    # all_rotation_candidates is the set the solver is ALLOWED to move through
    # (parent_adapter -> allowed_rots); it MUST be narrowed too, or the solver
    # rotates the connector back inward for packing. Regression guard.
    assert spec.all_rotation_candidates == [kept]


def test_filter_noop_without_detectable_opening():
    spec = _spec_with_connector(opening=None, edge="bottom")
    _filter_rotations_for_connector_opening(spec, _LOGGER)
    # Undetectable mouth: leave every candidate (placed as before).
    assert spec.rotation_candidates == [0.0, 90.0, 180.0, 270.0]
    assert spec.all_rotation_candidates == [0.0, 90.0, 180.0, 270.0]


def test_filter_keeps_all_when_unsatisfiable(caplog):
    # Two connectors pinned to opposite edges cannot both face out under one
    # rigid leaf rotation -> keep all + warn.
    c_bottom = AttachmentConstraint(
        ref="J1", target="edge", value="bottom", inward_keep_in_mm=0.0,
        outward_overhang_mm=0.5, source="child_artifact", child_index=0,
    )
    c_top = AttachmentConstraint(
        ref="J2", target="edge", value="top", inward_keep_in_mm=0.0,
        outward_overhang_mm=0.5, source="child_artifact", child_index=0,
    )
    candidates = [0.0, 90.0, 180.0, 270.0]

    def _conn(ref, rot):
        return Component(
            ref=ref, value="USB-C", pos=Point(0.0, 0.0), rotation=rot,
            layer=Layer.FRONT, width_mm=9.0, height_mm=3.0, kind="connector",
            pads=[], opening_direction=90.0,
        )

    models = {
        rot: SimpleNamespace(
            transformed=SimpleNamespace(
                transformed_components={"J1": _conn("J1", rot), "J2": _conn("J2", rot)}
            )
        )
        for rot in candidates
    }
    spec = SimpleNamespace(
        constraints=[c_bottom, c_top],
        rotation_candidates=list(candidates),
        all_rotation_candidates=list(candidates),
        models=models,
        instance_path="/leaf",
    )
    with caplog.at_level(logging.WARNING):
        _filter_rotations_for_connector_opening(spec, _LOGGER)
    assert spec.rotation_candidates == candidates
    assert spec.all_rotation_candidates == candidates
    assert any("no rotation places every edge-zoned part" in r.message
               for r in caplog.records)


def test_filter_keeps_only_rotations_with_zoned_part_at_extremity():
    """A mouthless edge-zoned part (e.g. a switch) must still be its leaf's
    extremity on the zoned side. Here SW1 is top-zoned and a sibling R1 sits
    above it at leaf-rotation 0/270 but below it at 180; only the rotations
    where SW1 is the topmost part survive (RC1 extremity criterion)."""
    constraint = AttachmentConstraint(
        ref="SW1", target="edge", value="top", inward_keep_in_mm=0.0,
        outward_overhang_mm=0.0, source="child_artifact", child_index=0,
    )

    def _part(ref, y):
        return Component(
            ref=ref, value=ref, pos=Point(0.0, y), rotation=0.0,
            layer=Layer.FRONT, width_mm=2.0, height_mm=2.0, kind="other", pads=[],
        )

    # Per rotation, place SW1 and a sibling R1; SW1 is topmost (smaller y) only
    # at rotations 90 and 180 in this stand-in.
    sw_y = {0.0: 5.0, 90.0: 0.0, 180.0: 0.0, 270.0: 5.0}
    r1_y = {0.0: 0.0, 90.0: 5.0, 180.0: 5.0, 270.0: 0.0}
    candidates = [0.0, 90.0, 180.0, 270.0]
    models = {
        rot: SimpleNamespace(
            transformed=SimpleNamespace(
                transformed_components={
                    "SW1": _part("SW1", sw_y[rot]),
                    "R1": _part("R1", r1_y[rot]),
                }
            )
        )
        for rot in candidates
    }
    spec = SimpleNamespace(
        constraints=[constraint],
        rotation_candidates=list(candidates),
        all_rotation_candidates=list(candidates),
        models=models,
        instance_path="/leaf",
    )
    _filter_rotations_for_connector_opening(spec, _LOGGER)
    assert spec.rotation_candidates == [90.0, 180.0]
    assert spec.all_rotation_candidates == [90.0, 180.0]


def test_filter_falls_back_to_mouth_correct_rotations(caplog):
    """When NO rotation satisfies mouth+extremity together, the filter must
    prefer the mouth-correct rotation(s) over giving up: the old all-candidates
    give-up let packing pick a rotation with the mouth 180deg INWARD -- an
    unmateable port at any outline (self-eval 2026-07-19 run_01: RV1 packed
    outboard of the BNC, so extremity was unsatisfiable at every rotation)."""
    constraint = AttachmentConstraint(
        ref="J1", target="edge", value="bottom", inward_keep_in_mm=0.0,
        outward_overhang_mm=0.5, source="child_artifact", child_index=0,
    )
    candidates = [0.0, 90.0, 180.0, 270.0]

    def _conn(rot):
        return Component(
            ref="J1", value="BNC", pos=Point(0.0, 0.0), rotation=rot,
            layer=Layer.FRONT, width_mm=9.0, height_mm=3.0, kind="connector",
            pads=[], opening_direction=90.0,
        )

    def _sibling():
        # Always OUTBOARD of J1 on the bottom side -> extremity never holds.
        return Component(
            ref="R1", value="R1", pos=Point(0.0, 30.0), rotation=0.0,
            layer=Layer.FRONT, width_mm=2.0, height_mm=2.0, kind="other",
            pads=[],
        )

    models = {
        rot: SimpleNamespace(
            transformed=SimpleNamespace(
                transformed_components={"J1": _conn(rot), "R1": _sibling()}
            )
        )
        for rot in candidates
    }
    spec = SimpleNamespace(
        constraints=[constraint],
        rotation_candidates=list(candidates),
        all_rotation_candidates=list(candidates),
        models=models,
        instance_path="/leaf",
    )
    with caplog.at_level(logging.WARNING):
        _filter_rotations_for_connector_opening(spec, _LOGGER)
    # opening 90 at comp rotation 0 -> board 90 == bottom outward: only 0.0.
    assert spec.rotation_candidates == [0.0]
    assert spec.all_rotation_candidates == [0.0]
    assert any("mouth-correct" in r.message for r in caplog.records)


# --- Layer C: overhang math ----------------------------------------------


def test_connector_edge_overhangs_not_insets():
    # outward_overhang_mm > 0, inward 0: the mouth anchor lands OUTSIDE the
    # outline so the port is proud of the FR4 (mateable). The legacy positive
    # inset put the anchor INSIDE the outline and buried the port.
    min_pt, max_pt = Point(0.0, 0.0), Point(20.0, 30.0)
    c = AttachmentConstraint(
        ref="J1", target="edge", value="bottom", inward_keep_in_mm=0.0,
        outward_overhang_mm=0.5, source="child_artifact", child_index=0,
    )
    target = edge_anchor_target_coordinate("bottom", c, min_pt, max_pt)
    assert target > max_pt.y  # mouth sits beyond the board bottom edge
    assert target == pytest.approx(max_pt.y + 0.5)
