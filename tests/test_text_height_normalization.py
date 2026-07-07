"""Silk text below the board's 0.8 mm min_text_height must be bumped at the
footprint-load seam (kicad_pcb_stub._normalize_text_heights) — the DRC class
that warned on essentially every board (self-eval 2026-07-07 quick win)."""

from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.design.synthesis.kicad_pcb_stub import _normalize_text_heights


def _fp_with_ref_size(h_mm: float, layer) -> object:
    board = pcbnew.CreateEmptyBoard()
    fp = pcbnew.FOOTPRINT(board)
    ref = fp.Reference()
    ref.SetLayer(layer)
    ref.SetTextSize(pcbnew.VECTOR2I(pcbnew.FromMM(h_mm), pcbnew.FromMM(h_mm)))
    ref.SetTextThickness(pcbnew.FromMM(0.105))
    return fp


def test_sub_min_silk_text_bumped_to_floor():
    fp = _fp_with_ref_size(0.7, pcbnew.F_SilkS)
    _normalize_text_heights(pcbnew, fp)
    size = fp.Reference().GetTextSize()
    assert pcbnew.ToMM(size.y) == pytest.approx(0.8, abs=1e-4)
    assert pcbnew.ToMM(fp.Reference().GetTextThickness()) >= 0.08


def test_fab_layer_text_untouched():
    fp = _fp_with_ref_size(0.7, pcbnew.F_Fab)
    _normalize_text_heights(pcbnew, fp)
    assert pcbnew.ToMM(fp.Reference().GetTextSize().y) == pytest.approx(0.7, abs=1e-4)


def test_legal_silk_text_untouched():
    fp = _fp_with_ref_size(1.0, pcbnew.F_SilkS)
    _normalize_text_heights(pcbnew, fp)
    assert pcbnew.ToMM(fp.Reference().GetTextSize().y) == pytest.approx(1.0, abs=1e-4)
