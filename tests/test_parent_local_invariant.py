"""Part 2 Levers 2.3/2.1: the shared body-pinned anchor + the parent-only-leaves
invariant guard.

- ``pad_centroid_anchor`` is the ONE body-pinned anchor formula (pad centroid ->
  body_center -> pos), shared by the leaf hole path and compose's parent-local
  path (previously duplicated inline).
- ``_warn_non_board_level_parent_local`` surfaces the parent-only-leaves
  invariant: a parent sheet should carry only leaves + board-level structure
  (mounting holes); any other loose component is flagged (the Lever 2.1
  auto-wrap target).
"""
from __future__ import annotations

import logging

from kicraft.autoplacer.brain.subcircuit_composer import pad_centroid_anchor
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point
from kicraft.cli.compose_subcircuits import _warn_non_board_level_parent_local


def _comp(ref, *, pads=None, body=None, pos=(0.0, 0.0)) -> Component:
    return Component(
        ref=ref, value="x", pos=Point(*pos), rotation=0.0, layer=Layer.FRONT,
        width_mm=3.0, height_mm=3.0, pads=pads or [], body_center=body,
    )


def test_pad_centroid_anchor_prefers_pad_centroid():
    c = _comp(
        "J1", pos=(0.0, 0.0), body=Point(9.0, 9.0),
        pads=[
            Pad(ref="J1", pad_id="1", pos=Point(2.0, 4.0), net="N", layer=Layer.FRONT),
            Pad(ref="J1", pad_id="2", pos=Point(4.0, 6.0), net="N", layer=Layer.FRONT),
        ],
    )
    assert pad_centroid_anchor(c) == Point(3.0, 5.0)  # centroid, not body/pos


def test_pad_centroid_anchor_falls_back_to_body_then_pos():
    assert pad_centroid_anchor(_comp("H1", body=Point(7.0, 8.0))) == Point(7.0, 8.0)
    assert pad_centroid_anchor(_comp("H2", pos=(1.0, 2.0))) == Point(1.0, 2.0)


def test_invariant_flags_non_board_level_components(caplog):
    # NB: _is_mounting_hole_ref recognizes only the "H<n>" form (not "MH").
    holes = {"H1": _comp("H1"), "H42": _comp("H42")}
    loose = {"J1": _comp("J1"), "U3": _comp("U3")}
    with caplog.at_level(logging.WARNING):
        assert _warn_non_board_level_parent_local({**holes, **loose}) == ["J1", "U3"]
    assert "parent-only-leaves invariant" in caplog.text


def test_invariant_silent_for_board_level_only(caplog):
    with caplog.at_level(logging.WARNING):
        assert _warn_non_board_level_parent_local({"H1": _comp("H1")}) == []
    assert caplog.text == ""
