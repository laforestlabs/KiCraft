"""WS8: circumscribe must not ship a massively oversized shaped board.

star-ornament shipped fab-ready at 592x563 mm (~63x its content area) because
``_fit_requested_shape`` circumscribed the shape around the content and never
consumed the brief's ``size_mm`` nor capped material overshoot.
"""

from __future__ import annotations

from kicraft.cli._compose_validate import (
    _MAX_SHAPE_AREA_RATIO,
    _requested_size_pair,
    _ring_area,
    _shape_fit_guard,
)


def test_ring_area_of_unit_square():
    assert _ring_area([(0, 0), (2, 0), (2, 2), (0, 2)]) == 4.0


def test_requested_size_pair_normalization():
    assert _requested_size_pair(60) == (60.0, 60.0)          # scalar diameter
    assert _requested_size_pair([80, 60]) == (80.0, 60.0)    # [w, h]
    assert _requested_size_pair({"w": 80, "h": 60}) == (80.0, 60.0)
    assert _requested_size_pair(None) is None
    assert _requested_size_pair("big") is None


def test_guard_rejects_area_explosion():
    # ~63x the content area with no size_mm -> rejected by the ratio cap.
    guard = _shape_fit_guard(
        "star", {}, content_area=2000.0,
        fitted_w=592.0, fitted_h=563.0, fitted_area=2000.0 * 63,
    )
    assert guard is not None
    assert guard["fitted"] is False
    assert guard["rejected_shape"] == "star"
    assert "content" in guard["reason"]


def test_guard_accepts_modest_convex_fit():
    # A circle ~1.6x its inscribed square -> under the cap, accepted (None).
    assert _shape_fit_guard(
        "circle", {}, content_area=1600.0,
        fitted_w=57.0, fitted_h=57.0, fitted_area=1600.0 * 1.6,
    ) is None


def test_guard_rejects_when_size_mm_exceeded():
    # Brief asked for Ø60; content forces a Ø90 circumscribe -> rejected.
    guard = _shape_fit_guard(
        "circle", {"size_mm": 60}, content_area=1600.0,
        fitted_w=90.0, fitted_h=90.0, fitted_area=1600.0 * 1.6,
    )
    assert guard is not None
    assert "size_mm" in guard["reason"]


def test_guard_honors_size_mm_within_slack():
    # Ø60 requested, Ø60.5 circumscribe (within 5% slack), small ratio -> accept.
    assert _shape_fit_guard(
        "circle", {"size_mm": 60}, content_area=2500.0,
        fitted_w=60.5, fitted_h=60.5, fitted_area=2500.0 * 1.2,
    ) is None


def test_cap_is_a_sane_default():
    assert 2.0 < _MAX_SHAPE_AREA_RATIO < 20.0
