"""Tests for placement_alignment: alignment-group detection + repair.

Locks the contract that:

* Two-member groups (the BT1/BT2 battery case) and N-member groups
  (LED rows, header arrays) go through the same code path.
* Mixed-value ic_groups are NOT picked up as alignment groups.
* The axis is derived from component_zones: zone:bottom -> row,
  edge:right -> column.
* Repair preserves the group's parallel-axis center -- it tightens
  alignment without dragging the group across the board.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.placement_alignment import (
    AlignmentGroup,
    apply_alignment_repair,
    detect_alignment_groups,
)
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point


def _comp(
    ref: str,
    *,
    pos: tuple[float, float],
    value: str = "18650",
    body_w: float = 18.0,
    body_h: float = 65.0,
) -> Component:
    return Component(
        ref=ref,
        value=value,
        pos=Point(*pos),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=body_w,
        height_mm=body_h,
        body_center=Point(*pos),
        kind="battery",
        is_through_hole=True,
        pads=[
            Pad(
                ref=ref,
                pad_id="1",
                pos=Point(pos[0], pos[1] - body_h / 2 + 1.0),
                net="VBAT",
                layer=Layer.FRONT,
                size_mm=Point(2.0, 2.0),
            ),
            Pad(
                ref=ref,
                pad_id="2",
                pos=Point(pos[0], pos[1] + body_h / 2 - 1.0),
                net="GND",
                layer=Layer.FRONT,
                size_mm=Point(2.0, 2.0),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_two_member_pair_with_zone_bottom_detects_row():
    components = {
        "BT1": _comp("BT1", pos=(20.0, 50.0)),
        "BT2": _comp("BT2", pos=(40.0, 51.5)),
    }
    cfg = {
        "ic_groups": {"BT1": ["BT2"]},
        "component_zones": {"BT1": {"zone": "bottom"}},
    }
    groups = detect_alignment_groups(cfg, components)
    assert len(groups) == 1
    g = groups[0]
    assert g.leader == "BT1"
    assert g.members == ("BT1", "BT2")
    assert g.axis == "row"
    assert g.parallel_axis == "x"
    assert g.perpendicular_axis == "y"
    # Pitch = max physical extent on parallel axis + 1mm clearance.
    # Body is 18 wide; pad bbox is 2x2 centered on body_center, so
    # physical bbox width is 18mm. Pitch = 19.0.
    assert g.pitch_mm == pytest.approx(19.0)


def test_three_member_row_for_led_strip_pattern():
    components = {
        "D1": _comp("D1", pos=(10.0, 5.0), value="LED_RED", body_w=3.0, body_h=2.0),
        "D2": _comp("D2", pos=(20.0, 5.5), value="LED_RED", body_w=3.0, body_h=2.0),
        "D3": _comp("D3", pos=(30.0, 4.7), value="LED_RED", body_w=3.0, body_h=2.0),
    }
    cfg = {
        "ic_groups": {"D1": ["D2", "D3"]},
        "component_zones": {"D1": {"zone": "top"}},
    }
    groups = detect_alignment_groups(cfg, components)
    assert len(groups) == 1
    assert groups[0].axis == "row"
    assert groups[0].members == ("D1", "D2", "D3")


def test_edge_right_detects_column():
    components = {
        "J2": _comp("J2", pos=(125.0, 20.0), value="HDR_3PIN", body_w=3.0, body_h=8.0),
        "J3": _comp("J3", pos=(125.5, 35.0), value="HDR_3PIN", body_w=3.0, body_h=8.0),
    }
    cfg = {
        "ic_groups": {"J2": ["J3"]},
        "component_zones": {"J2": {"edge": "right"}},
    }
    groups = detect_alignment_groups(cfg, components)
    assert len(groups) == 1
    assert groups[0].axis == "column"
    assert groups[0].parallel_axis == "y"


def test_mixed_value_group_is_rejected():
    components = {
        "U1": _comp("U1", pos=(10.0, 10.0), value="LM7805"),
        "C1": _comp("C1", pos=(15.0, 10.0), value="100nF"),
    }
    cfg = {
        "ic_groups": {"U1": ["C1"]},
        "component_zones": {"U1": {"zone": "bottom"}},
    }
    assert detect_alignment_groups(cfg, components) == []


def test_position_based_axis_inference_without_zone():
    """When components are clearly arranged on one axis already (small
    cross-axis spread, larger along-axis spread), the axis is inferred
    from positions alone -- no zone hint required."""
    components = {
        "BT1": _comp("BT1", pos=(20.0, 50.0)),
        "BT2": _comp("BT2", pos=(40.0, 50.5)),
    }
    cfg = {"ic_groups": {"BT1": ["BT2"]}}
    groups = detect_alignment_groups(cfg, components)
    assert len(groups) == 1
    assert groups[0].axis == "row"


def test_pre_solve_stacked_at_origin_falls_back_to_zone():
    """When all members sit at the same point (e.g. before the placer
    has run) position-based inference is ambiguous; zone hint wins."""
    components = {
        "BT1": _comp("BT1", pos=(0.0, 0.0)),
        "BT2": _comp("BT2", pos=(0.0, 0.0)),
    }
    cfg = {
        "ic_groups": {"BT1": ["BT2"]},
        "component_zones": {"BT1": {"zone": "bottom"}},
    }
    groups = detect_alignment_groups(cfg, components)
    assert len(groups) == 1
    assert groups[0].axis == "row"


def test_corner_zone_blocks_alignment_even_when_positions_align():
    """Two mounting holes at opposite corners share one axis incidentally
    (e.g. top-left at (2, 2) and top-right at (90, 2) share Y). Without
    the corner-zone block, position-based inference would treat them as
    a row and force them to share Y -- yanking them off their corners.
    The corner zone is the user saying "each lives at its own corner";
    detection must skip the group entirely."""
    components = {
        "H1": _comp("H1", pos=(2.0, 2.0), value="MOUNT_M3"),
        "H2": _comp("H2", pos=(50.0, 2.0), value="MOUNT_M3"),
    }
    cfg = {
        "ic_groups": {"H1": ["H2"]},
        "component_zones": {"H1": {"corner": "top-left"}},
    }
    assert detect_alignment_groups(cfg, components) == []


def test_no_zone_and_ambiguous_positions_skips():
    """When positions are equally spread on both axes (no clear linear
    arrangement) and there's no zone hint, the group is skipped --
    we'd rather no-op than guess wrong."""
    components = {
        "BT1": _comp("BT1", pos=(10.0, 10.0)),
        "BT2": _comp("BT2", pos=(30.0, 30.0)),  # same delta on both axes
    }
    cfg = {"ic_groups": {"BT1": ["BT2"]}}
    assert detect_alignment_groups(cfg, components) == []


def test_explicit_pitch_override_via_cfg():
    components = {
        "BT1": _comp("BT1", pos=(20.0, 50.0)),
        "BT2": _comp("BT2", pos=(40.0, 51.5)),
    }
    cfg = {
        "ic_groups": {"BT1": ["BT2"]},
        "component_zones": {"BT1": {"zone": "bottom"}},
        "alignment_pitch_mm": 30.0,
    }
    groups = detect_alignment_groups(cfg, components)
    assert groups[0].pitch_mm == 30.0


def test_missing_member_skips_group():
    components = {"BT1": _comp("BT1", pos=(20.0, 50.0))}
    cfg = {
        "ic_groups": {"BT1": ["BT2"]},  # BT2 not in components
        "component_zones": {"BT1": {"zone": "bottom"}},
    }
    assert detect_alignment_groups(cfg, components) == []


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def test_repair_snaps_pair_onto_shared_y_and_fixed_pitch():
    """The BT1/BT2 case: SA leaves them at slightly different Y values
    and irregular X spacing. Repair snaps Y to the mean and redistributes
    X at the configured pitch around the group's X center."""
    components = {
        "BT1": _comp("BT1", pos=(20.0, 50.0)),
        "BT2": _comp("BT2", pos=(45.0, 53.0)),
    }
    group = AlignmentGroup(
        leader="BT1",
        members=("BT1", "BT2"),
        axis="row",
        pitch_mm=20.0,
    )
    apply_alignment_repair(components, [group])

    # Mean Y was (50 + 53)/2 = 51.5 -> both members snap there.
    assert components["BT1"].pos.y == pytest.approx(51.5)
    assert components["BT2"].pos.y == pytest.approx(51.5)
    # Mean X was 32.5 -> centered: BT1 at 32.5 - 10 = 22.5, BT2 at 42.5.
    assert components["BT1"].pos.x == pytest.approx(22.5)
    assert components["BT2"].pos.x == pytest.approx(42.5)


def test_repair_preserves_group_parallel_center():
    """Group should land where SA put it on the parallel axis, just
    with internal alignment fixed."""
    components = {
        "BT1": _comp("BT1", pos=(60.0, 50.0)),
        "BT2": _comp("BT2", pos=(80.0, 51.0)),
    }
    group = AlignmentGroup(
        leader="BT1",
        members=("BT1", "BT2"),
        axis="row",
        pitch_mm=20.0,
    )
    apply_alignment_repair(components, [group])
    parallel_center = (components["BT1"].pos.x + components["BT2"].pos.x) / 2.0
    assert parallel_center == pytest.approx(70.0)


def test_repair_carries_pads_and_body_center_with_translation():
    components = {
        "BT1": _comp("BT1", pos=(20.0, 50.0)),
        "BT2": _comp("BT2", pos=(45.0, 53.0)),
    }
    group = AlignmentGroup(
        leader="BT1",
        members=("BT1", "BT2"),
        axis="row",
        pitch_mm=20.0,
    )
    bt1_pad1_offset = (
        components["BT1"].pads[0].pos.x - components["BT1"].pos.x,
        components["BT1"].pads[0].pos.y - components["BT1"].pos.y,
    )
    apply_alignment_repair(components, [group])
    new_pad1 = components["BT1"].pads[0].pos
    new_pos = components["BT1"].pos
    assert new_pad1.x == pytest.approx(new_pos.x + bt1_pad1_offset[0])
    assert new_pad1.y == pytest.approx(new_pos.y + bt1_pad1_offset[1])
    assert components["BT1"].body_center.x == pytest.approx(new_pos.x)
    assert components["BT1"].body_center.y == pytest.approx(new_pos.y)


def test_repair_three_members_evenly_spaced():
    components = {
        "D1": _comp("D1", pos=(10.0, 5.5), value="LED", body_w=3.0, body_h=2.0),
        "D2": _comp("D2", pos=(15.5, 5.0), value="LED", body_w=3.0, body_h=2.0),
        "D3": _comp("D3", pos=(21.0, 5.7), value="LED", body_w=3.0, body_h=2.0),
    }
    group = AlignmentGroup(
        leader="D1",
        members=("D1", "D2", "D3"),
        axis="row",
        pitch_mm=5.5,
    )
    apply_alignment_repair(components, [group])
    # Center X = (10 + 15.5 + 21) / 3 = 15.5; offsets -5.5, 0, +5.5.
    assert components["D1"].pos.x == pytest.approx(10.0)
    assert components["D2"].pos.x == pytest.approx(15.5)
    assert components["D3"].pos.x == pytest.approx(21.0)
    # All three share the mean Y.
    mean_y = (5.5 + 5.0 + 5.7) / 3.0
    for ref in ("D1", "D2", "D3"):
        assert components[ref].pos.y == pytest.approx(mean_y)


def test_repair_column_axis_swaps_role_of_x_and_y():
    components = {
        "J2": _comp("J2", pos=(125.0, 20.0), value="HDR", body_w=3.0, body_h=8.0),
        "J3": _comp("J3", pos=(125.5, 35.5), value="HDR", body_w=3.0, body_h=8.0),
    }
    group = AlignmentGroup(
        leader="J2",
        members=("J2", "J3"),
        axis="column",
        pitch_mm=10.0,
    )
    apply_alignment_repair(components, [group])
    # Column case: members share X (mean of 125.0 and 125.5 = 125.25),
    # distributed on Y at pitch 10 around current Y center 27.75.
    assert components["J2"].pos.x == pytest.approx(125.25)
    assert components["J3"].pos.x == pytest.approx(125.25)
    assert components["J2"].pos.y == pytest.approx(22.75)
    assert components["J3"].pos.y == pytest.approx(32.75)


def test_repair_no_groups_is_noop():
    components = {"BT1": _comp("BT1", pos=(20.0, 50.0))}
    snap = (components["BT1"].pos.x, components["BT1"].pos.y)
    apply_alignment_repair(components, [])
    assert (components["BT1"].pos.x, components["BT1"].pos.y) == snap


def test_repair_skips_group_with_only_one_member_present():
    """Defensive: if some members went missing between detection and
    repair, just skip the group rather than crash or single-point pin."""
    components = {"BT1": _comp("BT1", pos=(20.0, 50.0))}
    group = AlignmentGroup(
        leader="BT1",
        members=("BT1", "BT2"),  # BT2 not in components
        axis="row",
        pitch_mm=20.0,
    )
    snap = (components["BT1"].pos.x, components["BT1"].pos.y)
    apply_alignment_repair(components, [group])
    assert (components["BT1"].pos.x, components["BT1"].pos.y) == snap
