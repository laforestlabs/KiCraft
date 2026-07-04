"""Vendoring footprint hygiene: courtyard must clear pads by >= clearance.

``ensure_courtyard_clears_pads`` grows a footprint's courtyard so it encloses
every pad by at least the copper-to-edge clearance, run when vendoring a new
part (``add-part``). ``repair_malformed_courtyard`` rebuilds a courtyard whose
drawn segments never form a closed area (the easyEDA two-collinear-fp_line
pattern), which otherwise degenerates every consumer's extent math to a
stroke-width sliver. Gated on pcbnew (the footprint IO is KiCad's).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.parts_library.footprint_courtyard import (  # noqa: E402
    courtyard_pad_clearance_mm,
    ensure_courtyard_clears_pads,
    malformed_courtyard_layers,
    repair_malformed_courtyard,
)

# A footprint whose pad copper pokes 0.1mm PAST the courtyard on the left:
# pad x in [-0.5, 0.5], courtyard x in [-0.4, 0.6] -> left clearance -0.1mm.
_FP_NAME = "TESTCONN"
_KICAD_MOD = """(footprint "TESTCONN"
  (version 20240108)
  (generator "test")
  (layer "F.Cu")
  (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu"))
  (fp_rect (start -0.4 -0.6) (end 0.6 0.6)
    (stroke (width 0.05) (type default)) (fill none) (layer "F.CrtYd"))
)
"""


def _pretty_with_testconn(tmp_path: Path) -> str:
    pretty = tmp_path / "lib.pretty"
    pretty.mkdir()
    (pretty / f"{_FP_NAME}.kicad_mod").write_text(_KICAD_MOD)
    return str(pretty)


def test_grows_courtyard_so_pads_clear(tmp_path):
    pretty = _pretty_with_testconn(tmp_path)

    # Original: a pad pokes outside the courtyard (negative clearance).
    before = courtyard_pad_clearance_mm(pcbnew.FootprintLoad(pretty, _FP_NAME))
    assert before is not None and before < 0.2

    # Grow + persist (a fresh footprint object; never re-reads Pads post-mutate).
    fp = pcbnew.FootprintLoad(pretty, _FP_NAME)
    assert ensure_courtyard_clears_pads(fp, min_clearance_mm=0.2) is True
    pcbnew.PCB_IO_KICAD_SEXPR().FootprintSave(pretty, fp)

    # Reload: the courtyard now clears every pad by >= 0.2mm.
    after = courtyard_pad_clearance_mm(pcbnew.FootprintLoad(pretty, _FP_NAME))
    assert after >= 0.2 - 1e-4, after


def test_already_clear_courtyard_is_untouched(tmp_path):
    pretty = _pretty_with_testconn(tmp_path)
    # First grow makes it clear; a second grow must be a no-op (idempotent).
    fp = pcbnew.FootprintLoad(pretty, _FP_NAME)
    ensure_courtyard_clears_pads(fp, min_clearance_mm=0.2)
    pcbnew.PCB_IO_KICAD_SEXPR().FootprintSave(pretty, fp)

    fp2 = pcbnew.FootprintLoad(pretty, _FP_NAME)
    assert ensure_courtyard_clears_pads(fp2, min_clearance_mm=0.2) is False


# The real-world malformed pattern (srd-05vdc-sl-c relay, easyeda2kicad
# export): the ONLY courtyard graphics are the same vertical edge drawn once
# in each direction -- no closed area, so KiCad's courtyard polygon collapses
# to a stroke-width sliver at x=-8.7 while the part's body (silk outline) and
# pads span ~19 x 15 mm. Pads and silk mirror the real footprint's extents.
_BAD_NAME = "BADRELAY"
_BAD_KICAD_MOD = """(footprint "BADRELAY"
  (version 20240108)
  (generator "test")
  (layer "F.Cu")
  (fp_line (start -8.70 7.80) (end -8.70 -7.80)
    (stroke (width 0.05) (type default)) (layer "F.CrtYd"))
  (fp_line (start -8.70 -7.80) (end -8.70 7.80)
    (stroke (width 0.05) (type default)) (layer "F.CrtYd"))
  (fp_line (start -8.70 -7.80) (end 10.50 -7.80)
    (stroke (width 0.12) (type default)) (layer "F.SilkS"))
  (fp_line (start 10.50 -7.80) (end 10.50 7.80)
    (stroke (width 0.12) (type default)) (layer "F.SilkS"))
  (fp_line (start 10.50 7.80) (end -8.70 7.80)
    (stroke (width 0.12) (type default)) (layer "F.SilkS"))
  (fp_line (start -8.70 7.80) (end -8.70 -7.80)
    (stroke (width 0.12) (type default)) (layer "F.SilkS"))
  (pad "1" thru_hole circle (at -7.30 6.20) (size 2.5 2.5) (drill 1.3)
    (layers "*.Cu" "*.Mask"))
  (pad "2" thru_hole circle (at 7.30 6.20) (size 2.5 2.5) (drill 1.3)
    (layers "*.Cu" "*.Mask"))
  (pad "3" thru_hole circle (at 7.30 -6.20) (size 2.5 2.5) (drill 1.3)
    (layers "*.Cu" "*.Mask"))
)
"""


def _pretty_with_badrelay(tmp_path: Path) -> str:
    pretty = tmp_path / "badlib.pretty"
    pretty.mkdir()
    (pretty / f"{_BAD_NAME}.kicad_mod").write_text(_BAD_KICAD_MOD)
    return str(pretty)


def _courtyard_bbox_mm(fp) -> tuple[float, float]:
    bb = fp.GetCourtyard(pcbnew.F_CrtYd).BBox()
    return pcbnew.ToMM(bb.GetWidth()), pcbnew.ToMM(bb.GetHeight())


def test_malformed_courtyard_is_detected():
    pretty_dir = None  # built per-test below to keep fixtures independent

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pretty_dir = _pretty_with_badrelay(Path(td))
        fp = pcbnew.FootprintLoad(pretty_dir, _BAD_NAME)
        assert malformed_courtyard_layers(fp) == [pcbnew.F_CrtYd]

        # The degenerate extent that motivated the repair: the courtyard bbox
        # is a stroke-width sliver, nothing like the ~19x15mm part.
        w, h = _courtyard_bbox_mm(fp)
        assert w < 0.2, (w, h)


def test_valid_courtyard_is_not_flagged(tmp_path):
    pretty = _pretty_with_testconn(tmp_path)
    fp = pcbnew.FootprintLoad(pretty, _FP_NAME)
    assert malformed_courtyard_layers(fp) == []
    assert repair_malformed_courtyard(fp) is False


def test_repair_rebuilds_courtyard_around_pads_and_body(tmp_path):
    pretty = _pretty_with_badrelay(tmp_path)

    fp = pcbnew.FootprintLoad(pretty, _BAD_NAME)
    assert repair_malformed_courtyard(fp) is True
    pcbnew.PCB_IO_KICAD_SEXPR().FootprintSave(pretty, fp)

    # Reload: the courtyard is now a valid closed area covering the part.
    fp2 = pcbnew.FootprintLoad(pretty, _BAD_NAME)
    assert malformed_courtyard_layers(fp2) == []
    w, h = _courtyard_bbox_mm(fp2)
    # Body silk spans 19.2 x 15.6; pads poke to y +/-7.45. The rebuilt
    # rectangle must cover both plus the margin.
    assert w >= 19.2, (w, h)
    assert h >= 15.6, (w, h)
    # And it clears the pads like any healthy vendored footprint.
    assert courtyard_pad_clearance_mm(fp2) >= 0.2 - 1e-4

    # Idempotent: a repaired courtyard is no longer malformed.
    assert repair_malformed_courtyard(fp2) is False
