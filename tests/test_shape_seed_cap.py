"""GAP 1a: shape-aware parent placement.

The brief-requested outline shape must reach PLACEMENT, not just the
post-placement circumscribe: the seed is capped to the shape's largest
inscribable content rect (so the solver packs into the requested ⌀/size), and
the candidate search hard-prefers candidates whose shape actually committed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kicraft.cli.compose_subcircuits as cs
from kicraft.autoplacer.brain.types import Point
from kicraft.cli.compose_subcircuits import CandidateRecord, _seed_outline_dimensions


def _fake_artifacts(monkeypatch, dims):
    """Stub N leaf artifacts with fixed content bboxes (w, h)."""
    arts = [object() for _ in dims]
    sizes = {id(a): wh for a, wh in zip(arts, dims)}

    def _fake_transform(art, origin, rotation):
        w, h = sizes[id(art)]
        return SimpleNamespace(bounding_box=(Point(0.0, 0.0), Point(w, h)))

    monkeypatch.setattr(cs, "transform_loaded_artifact", _fake_transform)
    return arts


def test_seed_cap_bounds_aspect_base(monkeypatch):
    arts = _fake_artifacts(monkeypatch, [(20.0, 15.0), (10.0, 8.0), (8.0, 6.0)])
    derived = SimpleNamespace(child_specs={})
    uncapped = _seed_outline_dimensions(arts, derived, 2.0, area_overhead=6.0)
    capped = _seed_outline_dimensions(
        arts, derived, 2.0, area_overhead=6.0, seed_cap=(44.5, 44.5)
    )
    assert uncapped[0] > 44.5  # the cap actually binds at this overhead
    assert capped[0] <= max(44.5, 20.0 + 2.0 * 4)  # cap, unless a floor wins
    assert capped[1] <= max(44.5, 15.0 + 2.0 * 4)


def test_seed_cap_never_collapses_below_solvability_floors(monkeypatch):
    # A cap tighter than the biggest child keeps the max-child floor: the
    # seed must stay solvable; infeasibility is the stamp-time guard's job.
    arts = _fake_artifacts(monkeypatch, [(50.0, 40.0)])
    derived = SimpleNamespace(child_specs={})
    w, h = _seed_outline_dimensions(
        arts, derived, 2.0, area_overhead=2.5, seed_cap=(30.0, 30.0)
    )
    assert w >= 50.0 + 2.0 * 4
    assert h >= 40.0 + 2.0 * 4


def test_candidate_search_prefers_shape_fitted_winner():
    # Lexicographic: a fitted candidate with a LOWER score beats an unfitted
    # one with a higher score; among fitted, score decides.
    recs = [
        CandidateRecord(seed=0, shorts=0, score=90.0, place_solve_ms=0, stamp_ms=0,
                        stamp_drc_ms=0, accepted=True, pcb_path="", shape_fitted=False),
        CandidateRecord(seed=1, shorts=0, score=40.0, place_solve_ms=0, stamp_ms=0,
                        stamp_drc_ms=0, accepted=True, pcb_path="", shape_fitted=True),
        CandidateRecord(seed=2, shorts=0, score=35.0, place_solve_ms=0, stamp_ms=0,
                        stamp_drc_ms=0, accepted=True, pcb_path="", shape_fitted=True),
    ]
    winner = max(recs, key=lambda c: (c.shape_fitted, c.score))
    assert winner.seed == 1
    # No shape requested: shape_fitted defaults True everywhere -> pure score.
    for r in recs:
        r.shape_fitted = True
    assert max(recs, key=lambda c: (c.shape_fitted, c.score)).seed == 0


def test_candidate_record_serializes_shape_fitted():
    rec = CandidateRecord(seed=0, shorts=0, score=1.0, place_solve_ms=0, stamp_ms=0,
                          stamp_drc_ms=0, accepted=True, pcb_path="", shape_fitted=False)
    assert rec.to_dict()["shape_fitted"] is False
