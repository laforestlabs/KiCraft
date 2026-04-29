"""Regression test for ``_compute_layout_bbox`` track-edge inflation.

The parent composer's compaction step has a fast bbox-disjoint early-out
(``_bbox_disjoint`` in ``compose_subcircuits.py``) that skips the deeper
trace-aware blocker check. If the leaf bbox underlying that early-out is
*centerline-only* but the deeper check is *edge-aware*, the two systems
disagree: two leaves' centerline bboxes can be marked disjoint while
their actual track copper overlaps by up to a full track width.

In practice this manifests as GND/VSYS shorts at the seam between two
neighboring leaves -- the tracks were stamped overlapping in
``parent_pre_freerouting.kicad_pcb`` and FreeRouting (which sees them as
locked pre-routes) cannot fix them.

These tests lock the contract that ``_compute_layout_bbox`` inflates
trace endpoints by ``trace.width_mm / 2`` (and via positions by their
radius), so the bbox reflects physical extent and the early-out is a
correct conservative approximation of the deeper check.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.subcircuit_instances import _compute_layout_bbox
from kicraft.autoplacer.brain.types import (
    InterfaceAnchor,
    Layer,
    Point,
    TraceSegment,
    Via,
)


def test_trace_bbox_includes_half_width_on_each_axis():
    """A horizontal 0.2mm-wide trace from (0,0) to (10,0) must yield a
    bbox of (-0.1, -0.1) to (10.1, 0.1) -- centerline ± width/2.
    Centerline-only would return ((0,0), (10,0)), missing the
    cross-axis inflation that physical copper occupies."""
    traces = [
        TraceSegment(
            start=Point(0.0, 0.0),
            end=Point(10.0, 0.0),
            layer=Layer.FRONT,
            net="GND",
            width_mm=0.2,
        )
    ]
    tl, br = _compute_layout_bbox({}, traces, [], [])
    assert tl.x == pytest.approx(-0.1)
    assert tl.y == pytest.approx(-0.1)
    assert br.x == pytest.approx(10.1)
    assert br.y == pytest.approx(0.1)


def test_two_parallel_traces_bbox_extends_past_both_endpoints():
    """Diagonal trace at (1,1)->(5,4), width 0.3mm. Bbox must inflate by
    0.15 on each axis -- (0.85, 0.85) to (5.15, 4.15)."""
    traces = [
        TraceSegment(
            start=Point(1.0, 1.0),
            end=Point(5.0, 4.0),
            layer=Layer.FRONT,
            net="VSYS",
            width_mm=0.3,
        )
    ]
    tl, br = _compute_layout_bbox({}, traces, [], [])
    assert tl.x == pytest.approx(0.85)
    assert tl.y == pytest.approx(0.85)
    assert br.x == pytest.approx(5.15)
    assert br.y == pytest.approx(4.15)


def test_via_bbox_inflated_by_radius():
    """A via at (2,3) with size 0.6mm must yield bbox (1.7, 2.7)-(2.3, 3.3),
    inflated by radius (size/2). Centerline-only would yield a zero-area
    bbox at (2,3)."""
    vias = [
        Via(pos=Point(2.0, 3.0), drill_mm=0.3, size_mm=0.6, net="GND")
    ]
    tl, br = _compute_layout_bbox({}, [], vias, [])
    assert tl.x == pytest.approx(1.7)
    assert tl.y == pytest.approx(2.7)
    assert br.x == pytest.approx(2.3)
    assert br.y == pytest.approx(3.3)


def test_anchor_bbox_centerline_unchanged():
    """Interface anchors are connection points (no copper extent) and
    must keep using their centerline position. This guards against an
    over-eager refactor that inflates anchors too."""
    anchors = [
        InterfaceAnchor(
            port_name="VBUS",
            pos=Point(5.0, 5.0),
            layer=Layer.FRONT,
            pad_ref=("J1", "1"),
        )
    ]
    tl, br = _compute_layout_bbox({}, [], [], anchors)
    assert tl.x == pytest.approx(5.0)
    assert tl.y == pytest.approx(5.0)
    assert br.x == pytest.approx(5.0)
    assert br.y == pytest.approx(5.0)


def test_two_leaves_with_edge_traces_bboxes_overlap_when_centerlines_touch():
    """The actual scenario from the LLUPS short bug. Leaf A has a trace
    along its bottom edge (centerline at y=10); Leaf B is offset above
    it by 10.0 mm so its top-edge trace centerline is at y=10. With
    edge-aware bboxes, A's bbox.y_max = 10.1 and B's bbox.y_min = 9.9 --
    they OVERLAP, which is correct: the physical tracks would short.

    Without the fix, both bboxes had y=10 exactly and were
    bbox-disjoint, so the compaction's early-out would skip the deeper
    blocker check and let the overlap through.
    """
    traces_a = [
        TraceSegment(
            start=Point(0.0, 10.0),
            end=Point(5.0, 10.0),
            layer=Layer.FRONT,
            net="GND",
            width_mm=0.2,
        )
    ]
    traces_b = [
        TraceSegment(
            start=Point(0.0, 10.0),
            end=Point(5.0, 10.0),
            layer=Layer.FRONT,
            net="VSYS",
            width_mm=0.2,
        )
    ]
    tl_a, br_a = _compute_layout_bbox({}, traces_a, [], [])
    tl_b, br_b = _compute_layout_bbox({}, traces_b, [], [])

    # Both bboxes inflate to y=[9.9, 10.1] -- they overlap, which is the
    # correct physical answer. Two centerline-only bboxes at y=10 each
    # would have br_a.y == tl_b.y == 10 and the disjoint test
    # (br_a.y <= tl_b.y) would mark them disjoint.
    assert br_a.y == pytest.approx(10.1)
    assert tl_b.y == pytest.approx(9.9)
    assert br_a.y > tl_b.y, (
        "edge-aware bboxes must overlap when centerlines coincide -- "
        "otherwise the compaction early-out misses real shorts"
    )


def test_zero_width_trace_does_not_inflate_to_negative():
    """Defensive: width_mm < 0 (corrupt data) must not produce a
    NEGATIVE half-width that shrinks the bbox below the centerline."""
    traces = [
        TraceSegment(
            start=Point(0.0, 0.0),
            end=Point(10.0, 0.0),
            layer=Layer.FRONT,
            net="?",
            width_mm=-1.0,
        )
    ]
    tl, br = _compute_layout_bbox({}, traces, [], [])
    # half_w clamped to 0; bbox uses raw centerlines.
    assert tl.x == pytest.approx(0.0)
    assert br.x == pytest.approx(10.0)
    assert tl.y == pytest.approx(0.0)
    assert br.y == pytest.approx(0.0)
