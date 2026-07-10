"""Mechanical-conformance check (PR3 foundation): geometry, not net names."""

from __future__ import annotations

from kicraft.form_factors import get_template
from kicraft.form_factors.conformance import (
    check_conformance,
    expected_pins,
    board_local_pads,
)


def _template():
    return get_template("arduino_uno_shield")


def test_exact_placement_is_conformant():
    t = _template()
    delivered = [(x, y) for _net, x, y in expected_pins(t)]
    r = check_conformance(t, delivered, (t.board_width_mm, t.board_height_mm))
    assert r.conformant is True
    assert r.matched_pins == r.total_pins == 32
    assert r.outline_ok is True


def test_free_placed_board_is_non_conformant():
    # KC-99A9M8-style: headers exist but nowhere near the standard positions,
    # and the board is the wrong size.
    t = _template()
    delivered = [(200.0 + i, 5.0) for i in range(32)]  # a column off in the weeds
    r = check_conformance(t, delivered, (185.0, 40.0))
    assert r.conformant is False
    assert r.matched_pins == 0
    assert r.outline_ok is False
    assert len(r.missing) == 32
    assert "NON-CONFORMANT" in r.summary()


def test_tolerance_band():
    t = _template()
    # Every pin shifted 1.0mm -> within the default 1.5mm tol.
    within = [(x + 1.0, y) for _n, x, y in expected_pins(t)]
    assert check_conformance(t, within, (t.board_width_mm, t.board_height_mm)).conformant
    # Shifted 3mm -> outside tol.
    outside = [(x + 3.0, y) for _n, x, y in expected_pins(t)]
    assert not check_conformance(t, outside, (t.board_width_mm, t.board_height_mm)).conformant


def test_outline_mismatch_alone_fails_conformance():
    t = _template()
    delivered = [(x, y) for _n, x, y in expected_pins(t)]
    r = check_conformance(t, delivered, (70.0, 55.0))  # ~2mm too big both axes
    assert r.matched_pins == r.total_pins  # pins fine
    assert r.outline_ok is False           # but outline wrong
    assert r.conformant is False


def test_board_local_pads_normalizes_and_reads(tmp_path):
    # A tiny board with a shifted Edge.Cuts origin + one footprint/pad.
    pcb = tmp_path / "b.kicad_pcb"
    pcb.write_text(
        """(kicad_pcb
  (gr_line (start 100 200) (end 110 200) (layer "Edge.Cuts"))
  (gr_line (start 100 200) (end 100 210) (layer "Edge.Cuts"))
  (gr_line (start 110 200) (end 110 210) (layer "Edge.Cuts"))
  (gr_line (start 100 210) (end 110 210) (layer "Edge.Cuts"))
  (footprint "x" (at 105 205)
    (pad "1" thru_hole circle (at 0 0) (size 1 1))
  )
)
"""
    )
    pads, wh = board_local_pads(str(pcb))
    assert wh == (10.0, 10.0)
    # Pad at world (105,205); Edge min corner (100,200) -> local (5,5).
    assert pads == [(5.0, 5.0)]


def test_board_local_pads_applies_footprint_rotation(tmp_path):
    # A header laid horizontally along an edge is stamped rotated 90 deg: its pad
    # locals advance +Y but the WORLD pads must advance +X. The reader must apply
    # the footprint rotation or a conformant board reads as non-conformant.
    pcb = tmp_path / "r.kicad_pcb"
    pcb.write_text(
        """(kicad_pcb
  (gr_line (start 0 0) (end 20 0) (layer "Edge.Cuts"))
  (gr_line (start 0 0) (end 0 20) (layer "Edge.Cuts"))
  (gr_line (start 20 0) (end 20 20) (layer "Edge.Cuts"))
  (gr_line (start 0 20) (end 20 20) (layer "Edge.Cuts"))
  (footprint "h" (at 5 5 90)
    (pad "1" thru_hole circle (at 0 0 90) (size 1 1))
    (pad "2" thru_hole circle (at 0 2.54 90) (size 1 1))
  )
)
"""
    )
    pads, _wh = board_local_pads(str(pcb))
    # pad1 at origin (5,5); pad2 local (0,2.54) rotated CW 90 -> (+2.54,0) world
    # (7.54,5). Without applying rotation it would wrongly read (5,7.54).
    assert (5.0, 5.0) in pads
    assert any(abs(px - 7.54) < 1e-6 and abs(py - 5.0) < 1e-6 for px, py in pads)
