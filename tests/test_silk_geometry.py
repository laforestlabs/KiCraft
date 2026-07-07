"""Pure-geometry tests for the silkscreen placement helpers."""

from kicraft.autoplacer.hardware.silk_geometry import (
    bbox_inside_poly,
    boxes_overlap,
    clamp_shift_into_bbox,
    find_shift_into_poly,
    point_in_poly,
    poly_bbox,
    shift_box,
)

RECT = [(0.0, 0.0), (50.0, 0.0), (50.0, 30.0), (0.0, 30.0)]
# Rounded-ish top-left corner (chamfer stands in for the arc): the corner
# region outside the chamfer is OFF the board even though it is inside the
# bounding box — exactly the KC-7A3VEX "PD TRIGGER" clip geometry.
CHAMFERED = [(6.0, 0.0), (50.0, 0.0), (50.0, 30.0), (0.0, 30.0), (0.0, 6.0)]


def test_point_in_poly_basic():
    assert point_in_poly(25, 15, RECT)
    assert not point_in_poly(-1, 15, RECT)
    assert not point_in_poly(25, 31, RECT)


def test_point_in_poly_chamfered_corner():
    assert point_in_poly(1.0, 1.0, RECT)
    assert not point_in_poly(1.0, 1.0, CHAMFERED)  # cut corner is off-board
    assert point_in_poly(10.0, 10.0, CHAMFERED)


def test_bbox_inside_poly_margin():
    assert bbox_inside_poly((1, 1, 10, 5), RECT)
    assert bbox_inside_poly((1, 1, 10, 5), RECT, margin=0.5)
    # margin pushes the probe outside
    assert not bbox_inside_poly((0.2, 1, 10, 5), RECT, margin=0.5)


def test_boxes_overlap_clearance():
    assert boxes_overlap((0, 0, 10, 10), (5, 5, 15, 15))
    assert not boxes_overlap((0, 0, 10, 10), (11, 0, 20, 10))
    # 1mm apart but 2mm clearance required -> counts as overlap
    assert boxes_overlap((0, 0, 10, 10), (11, 0, 20, 10), clearance=2.0)


def test_clamp_shift_into_bbox():
    bound = poly_bbox(RECT)
    dx, dy = clamp_shift_into_bbox((-3, 5, 7, 9), bound, margin=1.0)
    assert dx == 4.0 and dy == 0.0
    dx, dy = clamp_shift_into_bbox((45, 28, 55, 33), bound, margin=1.0)
    assert dx == -6.0 and dy == -4.0


def test_find_shift_pulls_clipped_label_inside():
    # A label hanging off the left edge (the observed clip class).
    box = (-4.0, 10.0, 8.0, 12.0)
    shift = find_shift_into_poly(box, RECT, margin=0.2)
    assert shift is not None
    assert bbox_inside_poly(shift_box(box, *shift), RECT, 0.2)


def test_find_shift_handles_rounded_corner():
    # In-bbox but crossing the chamfered corner: the bbox clamp alone is not
    # enough; the ring search must move it away from the cut.
    box = (0.5, 0.5, 9.5, 2.5)
    assert not bbox_inside_poly(box, CHAMFERED, 0.2)
    shift = find_shift_into_poly(box, CHAMFERED, margin=0.2)
    assert shift is not None
    assert bbox_inside_poly(shift_box(box, *shift), CHAMFERED, 0.2)


def test_find_shift_gives_up_on_oversized_label():
    box = (0.0, 0.0, 80.0, 40.0)  # bigger than the board
    assert find_shift_into_poly(box, RECT, margin=0.2) is None
