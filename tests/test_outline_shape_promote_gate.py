"""Outline-shape promote gate: `_check_outline_shape_conformance`.

A brief-requested outline shape ("round 60 mm") must be DELIVERED: a rect
fallback or a wrong-size shape fails the fab gate loudly instead of shipping
as silently fab-ready (KC-HN59RJ). Shapes the pipeline has no generator for
are advisory (enforced=False); validated standards are PR3's job (None).
"""

from __future__ import annotations

import math
from pathlib import Path

from kicraft.design import cli_app
from kicraft.design.models import ConversationState, FormFactor, IntentSlot


def _state(shape: str | None, size_mm: float | None = None, standard: str | None = None):
    ff = FormFactor(shape=shape or "rect", size_mm=size_mm, standard=standard)
    return ConversationState(project_stem="X", intent=IntentSlot(goal="g", form_factor=ff))


def _rect_pcb(tmp_path: Path, w: float = 53.0, h: float = 38.0) -> Path:
    pcb = tmp_path / "rect.kicad_pcb"
    pcb.write_text(
        "(kicad_pcb\n"
        f'  (gr_line (start 0 0) (end {w} 0) (layer "Edge.Cuts"))\n'
        f'  (gr_line (start {w} 0) (end {w} {h}) (layer "Edge.Cuts"))\n'
        f'  (gr_line (start {w} {h}) (end 0 {h}) (layer "Edge.Cuts"))\n'
        f'  (gr_line (start 0 {h}) (end 0 0) (layer "Edge.Cuts"))\n'
        '  (gr_line (start 0 0) (end 10 0) (layer "F.Silkscreen"))\n'
        ")\n"
    )
    return pcb


def _circle_pcb(tmp_path: Path, diameter_mm: float = 60.0, segments: int = 32) -> Path:
    r = diameter_mm / 2.0
    cx = cy = r + 1.0
    pts = [
        (cx + r * math.cos(2 * math.pi * i / segments),
         cy + r * math.sin(2 * math.pi * i / segments))
        for i in range(segments)
    ]
    lines = ["(kicad_pcb"]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        lines.append(
            f'  (gr_line (start {x0:.4f} {y0:.4f}) (end {x1:.4f} {y1:.4f})'
            f' (layer "Edge.Cuts"))'
        )
    lines.append(")")
    pcb = tmp_path / f"circle_{int(diameter_mm)}.kicad_pcb"
    pcb.write_text("\n".join(lines))
    return pcb


def test_none_for_rect_or_absent_shape(tmp_path):
    assert cli_app._check_outline_shape_conformance(_state("rect"), _rect_pcb(tmp_path)) is None
    assert cli_app._check_outline_shape_conformance(_state(None), _rect_pcb(tmp_path)) is None


def test_rect_fallback_fails_loudly(tmp_path):
    res = cli_app._check_outline_shape_conformance(
        _state("circle", 60.0), _rect_pcb(tmp_path)
    )
    assert res is not None
    assert res["conformant"] is False
    assert res["enforced"] is True  # circle IS supported -> hard gate
    assert "RECT FALLBACK" in res["summary"]


def test_delivered_circle_at_size_conforms(tmp_path):
    res = cli_app._check_outline_shape_conformance(
        _state("circle", 60.0), _circle_pcb(tmp_path, 60.0)
    )
    assert res is not None
    assert res["conformant"] is True, res["summary"]


def test_wrong_size_circle_fails(tmp_path):
    res = cli_app._check_outline_shape_conformance(
        _state("circle", 60.0), _circle_pcb(tmp_path, 45.0)
    )
    assert res is not None
    assert res["conformant"] is False
    assert "SIZE MISMATCH" in res["summary"]


def test_shape_without_size_checks_shape_only(tmp_path):
    res = cli_app._check_outline_shape_conformance(
        _state("circle"), _circle_pcb(tmp_path, 45.0)
    )
    assert res is not None and res["conformant"] is True


def test_unsupported_shape_is_advisory(tmp_path):
    res = cli_app._check_outline_shape_conformance(
        _state("gear", 60.0), _rect_pcb(tmp_path)
    )
    assert res is not None
    assert res["conformant"] is False
    assert res["enforced"] is False  # no generator -> advisory, not a hard fail


def test_validated_standard_defers_to_pr3_gate(tmp_path):
    res = cli_app._check_outline_shape_conformance(
        _state("circle", 60.0, standard="arduino_uno_shield"), _rect_pcb(tmp_path)
    )
    assert res is None
