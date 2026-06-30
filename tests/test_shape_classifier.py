"""Phase 6 (foundation): deterministic Edge.Cuts shape classifier.

Round-trips every outline through the SAME polyline generators the stamper uses
(OutlineSpec for parametric, kicraft.shapes for named) and asserts the
classifier reports the right coarse family. This is the $0 deterministic check
the self-eval uses to confirm a built board matches the requested form factor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kicraft.autoplacer.brain.types import Point
from kicraft.layout_editor.outline import OutlineSpec
from kicraft.layout_editor.outline import circumscribe as circ_param
from kicraft.render.edge_cuts import (
    classify_edge_cuts_shape,
    classify_ring,
    family_for_shape,
)
from kicraft.shapes import circumscribe as circ_poly

TL, BR = Point(0.0, 0.0), Point(40.0, 20.0)


def _param_ring(shape, **kw):
    spec = circ_param(shape, TL, BR, **kw) if shape != "rect" else OutlineSpec.rect(TL, BR)
    return [(p.x, p.y) for p in spec.polyline()]


def _poly_ring(name):
    return list(circ_poly(name, SimpleNamespace(x=0.0, y=0.0), SimpleNamespace(x=40.0, y=20.0)).points())


# --------------------------------------------------------------------------- #
# Parametric shapes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape,family", [
    ("rect", "rectangular"),
    ("circle", "round"),
    ("rounded_rect", "round"),
    ("chamfered_rect", "polygon"),
])
def test_classify_parametric(shape, family):
    assert classify_ring(_param_ring(shape))["family"] == family


# --------------------------------------------------------------------------- #
# Named / compound shapes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,family", [
    ("triangle", "polygon"),
    ("pentagon", "polygon"),
    ("hexagon", "polygon"),
    ("octagon", "polygon"),
    ("star", "star"),
    ("snowman", "compound"),
])
def test_classify_named(name, family):
    assert classify_ring(_poly_ring(name))["family"] == family


def test_hexagon_reports_six_corners():
    res = classify_ring(_poly_ring("hexagon"))
    assert res["label"] == "hexagon"
    assert res["n_corners"] == 6


def test_named_shapes_are_non_rectangular():
    # The headline guarantee: any requested shape produces a non-rectangular
    # board (the rubric's hard signal).
    for name in ("circle", "hexagon", "star", "snowman", "heart"):
        ring = _param_ring("circle") if name == "circle" else _poly_ring(name)
        assert classify_ring(ring)["family"] != "rectangular", name


# --------------------------------------------------------------------------- #
# family_for_shape mapping + file parser
# --------------------------------------------------------------------------- #

def test_family_for_shape_mapping():
    assert family_for_shape("circle") == "round"
    assert family_for_shape("Hexagon") == "polygon"
    assert family_for_shape("snowman") == "compound"
    assert family_for_shape("rect") == "rectangular"


def _write_pcb(tmp_path, ring):
    lines = []
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        lines.append(
            f'  (gr_line (start {x0} {y0}) (end {x1} {y1}) '
            f'(stroke (width 0.1) (type solid)) (layer "Edge.Cuts"))'
        )
    pcb = tmp_path / "b.kicad_pcb"
    pcb.write_text("(kicad_pcb\n" + "\n".join(lines) + "\n)\n", encoding="utf-8")
    return pcb


def test_classify_from_kicad_pcb_file(tmp_path):
    pcb = _write_pcb(tmp_path, _poly_ring("hexagon"))
    res = classify_edge_cuts_shape(pcb)
    assert res is not None
    assert res["family"] == "polygon"


def test_classify_missing_file_is_none(tmp_path):
    assert classify_edge_cuts_shape(tmp_path / "nope.kicad_pcb") is None
