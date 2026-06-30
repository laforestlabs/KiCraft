"""Phase 6: deterministic outline-shape grading for the self-eval."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kicraft.autoplacer.brain.types import Point
from kicraft.eval.outline_check import evaluate_outline_shape
from kicraft.layout_editor.outline import OutlineSpec
from kicraft.layout_editor.outline import circumscribe as circ_param
from kicraft.shapes import circumscribe as circ_poly
from kicraft.tuning.benchmark import SHAPED_OUTLINE_PROMPTS

TL, BR = Point(0.0, 0.0), Point(40.0, 20.0)


def _ring(shape):
    if shape == "rect":
        return [(p.x, p.y) for p in OutlineSpec.rect(TL, BR).polyline()]
    try:
        return [(p.x, p.y) for p in circ_param(shape, TL, BR).polyline()]
    except ValueError:
        return list(circ_poly(shape, SimpleNamespace(x=0.0, y=0.0),
                              SimpleNamespace(x=40.0, y=20.0)).points())


def _pcb(tmp_path, shape):
    ring = _ring(shape)
    lines = [
        f'  (gr_line (start {x0} {y0}) (end {x1} {y1}) (layer "Edge.Cuts"))'
        for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1])
    ]
    pcb = tmp_path / f"{shape}.kicad_pcb"
    pcb.write_text("(kicad_pcb\n" + "\n".join(lines) + "\n)\n", encoding="utf-8")
    return pcb


@pytest.mark.parametrize("shape", ["circle", "rounded_rect", "hexagon", "star", "snowman"])
def test_matching_shape_scores_top(tmp_path, shape):
    res = evaluate_outline_shape(_pcb(tmp_path, shape), shape)
    assert res["level"] == 4, res


def test_rectangular_when_shape_requested_scores_zero(tmp_path):
    res = evaluate_outline_shape(_pcb(tmp_path, "rect"), "circle")
    assert res["level"] == 0


def test_wrong_family_but_non_rect_scores_partial(tmp_path):
    # Built a circle, asked for a hexagon: it took a shape, just not that one.
    res = evaluate_outline_shape(_pcb(tmp_path, "circle"), "hexagon")
    assert res["level"] == 2
    assert res["detected_family"] == "round"


def test_no_shape_requested_is_not_applicable(tmp_path):
    res = evaluate_outline_shape(_pcb(tmp_path, "rect"), "rect")
    assert res["level"] is None
    assert res["partial"] is True


def test_missing_pcb_scores_zero():
    res = evaluate_outline_shape("/no/such/board.kicad_pcb", "circle")
    assert res["level"] == 0
    assert res["detected_family"] is None


def test_corpus_entries_are_well_formed():
    from kicraft.layout_editor.outline import SHAPES
    from kicraft.shapes import KNOWN_SHAPES

    known = set(SHAPES) | set(KNOWN_SHAPES)
    slugs = set()
    for e in SHAPED_OUTLINE_PROMPTS:
        assert e["slug"] not in slugs, f"duplicate slug {e['slug']}"
        slugs.add(e["slug"])
        assert e["outline_shape"] in known, f"{e['slug']}: unbuildable shape {e['outline_shape']}"
        assert e["brief"].strip()
