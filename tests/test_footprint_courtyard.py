"""Vendoring footprint hygiene: courtyard must clear pads by >= clearance.

``ensure_courtyard_clears_pads`` grows a footprint's courtyard so it encloses
every pad by at least the copper-to-edge clearance, run when vendoring a new
part (``add-part``). Gated on pcbnew (the footprint IO is KiCad's).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.parts_library.footprint_courtyard import (  # noqa: E402
    courtyard_pad_clearance_mm,
    ensure_courtyard_clears_pads,
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
