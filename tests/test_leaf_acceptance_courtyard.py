"""Tests for the leaf-acceptance courtyard-overlap gate.

A leaf placement is stamped rigidly into the parent, so a same-side courtyard
overlap on the leaf becomes a terminal ``courtyards_overlap`` rc7. The
``no_gross_courtyard_overlap`` gate surfaces it AT THE LEAF -- with the same
minor-clip tolerance the terminal fab gate uses -- so the solver can re-place
instead of shipping a doomed leaf. Regression guard for KC-U2VAA8 (two D12.5mm
radial caps whose pin-1 origins offset their bodies into an 8mm overlap).

Synthetic data only; ``measure_courtyard_overlaps`` is monkeypatched so the
test needs no pcbnew or on-disk board.
"""

from __future__ import annotations

import kicraft.autoplacer.courtyard_overlap as co_mod
from kicraft.autoplacer.brain.leaf_acceptance import (
    LeafAcceptanceConfig,
    _gate_no_gross_courtyard_overlap,
    evaluate_leaf_acceptance,
)
from kicraft.autoplacer.courtyard_overlap import CourtyardOverlap


def _co(pen: float, area: float, ra: str = "C1", rb: str = "C2") -> CourtyardOverlap:
    return CourtyardOverlap(
        ref_a=ra, ref_b=rb, layer="B", area_mm2=area, penetration_mm=pen
    )


def _validation(courtyard: int, board_path: str = "/x/board.kicad_pcb") -> dict:
    return {"drc": {"courtyard": courtyard}, "board_path": board_path}


def test_no_courtyard_overlap_passes():
    ok, detail = _gate_no_gross_courtyard_overlap(
        _validation(0), {}, LeafAcceptanceConfig()
    )
    assert ok
    assert detail["passed"]
    assert detail["courtyard_overlaps"] == 0


def test_gross_overlap_fails(monkeypatch):
    monkeypatch.setattr(co_mod, "measure_courtyard_overlaps", lambda _p: [_co(8.0, 69.0)])
    ok, detail = _gate_no_gross_courtyard_overlap(
        _validation(1), {}, LeafAcceptanceConfig()
    )
    assert not ok
    assert detail["gross"]
    assert not detail["minor"]


def test_minor_clip_is_tolerated(monkeypatch):
    # Below both thresholds (0.5mm / 0.5mm^2) -> assemblable -> pass, matching
    # the terminal fab gate's minor-clip tolerance.
    monkeypatch.setattr(co_mod, "measure_courtyard_overlaps", lambda _p: [_co(0.1, 0.1)])
    ok, detail = _gate_no_gross_courtyard_overlap(
        _validation(1), {}, LeafAcceptanceConfig()
    )
    assert ok
    assert detail["minor"]
    assert not detail["gross"]


def test_unmeasurable_overlap_is_conservative(monkeypatch):
    # DRC flagged an overlap but pcbnew could not measure it -> hard-fail rather
    # than mis-grade it minor (mirrors the module's documented contract).
    monkeypatch.setattr(co_mod, "measure_courtyard_overlaps", lambda _p: [])
    ok, detail = _gate_no_gross_courtyard_overlap(
        _validation(1), {}, LeafAcceptanceConfig()
    )
    assert not ok
    assert detail["unmeasurable"]


def test_gate_can_be_disabled(monkeypatch):
    monkeypatch.setattr(co_mod, "measure_courtyard_overlaps", lambda _p: [_co(8.0, 69.0)])
    cfg = LeafAcceptanceConfig(require_no_gross_courtyard_overlap=False)
    ok, detail = _gate_no_gross_courtyard_overlap(_validation(1), {}, cfg)
    assert ok
    assert detail.get("skipped")


def test_evaluate_leaf_acceptance_rejects_gross_overlap(monkeypatch):
    monkeypatch.setattr(co_mod, "measure_courtyard_overlaps", lambda _p: [_co(8.0, 69.0)])
    validation = {
        "board_exists": True,
        "board_path": "/x/board.kicad_pcb",
        "drc": {"shorts": 0, "unconnected": 0, "clearance": 0, "courtyard": 1},
        "track_summary": {"traces": 5, "vias": 0},
    }
    result = evaluate_leaf_acceptance(validation, {}, LeafAcceptanceConfig())
    assert not result.accepted
    assert "no_gross_courtyard_overlap" in result.rejection_reasons
    # The condensed DRC summary now carries the courtyard count too.
    assert result.drc_summary["courtyard"] == 1


def test_evaluate_leaf_acceptance_accepts_minor_clip(monkeypatch):
    monkeypatch.setattr(co_mod, "measure_courtyard_overlaps", lambda _p: [_co(0.1, 0.1)])
    validation = {
        "board_exists": True,
        "board_path": "/x/board.kicad_pcb",
        "drc": {"shorts": 0, "unconnected": 0, "clearance": 0, "courtyard": 1},
        "track_summary": {"traces": 5, "vias": 0},
    }
    result = evaluate_leaf_acceptance(validation, {}, LeafAcceptanceConfig())
    assert "no_gross_courtyard_overlap" not in result.rejection_reasons
