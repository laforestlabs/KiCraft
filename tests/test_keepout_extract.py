"""Tests for hardware.keepout_extract: preserve + inject + rotated transform."""
from __future__ import annotations

import os

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.config import DEFAULT_CONFIG  # noqa: E402
from kicraft.autoplacer.hardware.keepout_extract import (  # noqa: E402
    _transform_local_rect,
    extract_keepout_rects,
)

_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kicraft",
    "parts_library",
)
_S3_PRETTY = os.path.join(_LIB, "esp32-s3-wroom-1", "esp32-s3-wroom-1.pretty")
_NAME = "WIRELM-SMD_ESP32-S3-WROOM-1"
# A genuinely keep-out-free footprint, for the inject-without-preserve case.
_SOT23_PRETTY = os.path.join(_LIB, "ao3400a", "ao3400a.pretty")
_SOT23_NAME = "SOT-23-3_L2.9-W1.3-P1.90-LS2.4-BR"


def _board_with_footprint(path, pretty, name, *, ref="U1", x=30.0, y=35.0, rot=0.0):
    """Build, save, and reload a board with one footprint at the given pose.

    Save+reload (the path adapter.load uses) yields properly-typed FOOTPRINT
    proxies; repeated in-memory NewBoard+Add(fp) makes board.Footprints() return
    raw SwigPyObjects across successive tests in one process.
    """
    mm = pcbnew.FromMM
    board = pcbnew.NewBoard(path)
    fp = pcbnew.FootprintLoad(pretty, name)
    if fp is None:
        pytest.skip(f"could not load {name}")
    fp.SetReference(ref)
    fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
    fp.SetOrientationDegrees(rot)
    board.Add(fp)
    board.BuildConnectivity()
    board.Save(path)
    return pcbnew.LoadBoard(path)


def _board_with_wroom(path, *, x=30.0, y=35.0, rot=0.0):
    return _board_with_footprint(path, _S3_PRETTY, _NAME, ref="U1", x=x, y=y, rot=rot)


def test_preserve_extracts_footprint_internal_keepout(tmp_path):
    # empty cfg -> only the preserve path fires
    board = _board_with_wroom(str(tmp_path / "b.kicad_pcb"))
    rects = extract_keepout_rects(board, {})
    preserve = [r for r in rects if r.source == "preserve"]
    assert len(preserve) == 1, f"expected 1 preserved internal keep-out, got {rects}"
    r = preserve[0]
    assert r.owner_ref == "U1"
    # antenna strip local x[-9,9] y[-16.4,-10] placed at (30,35), rot 0
    assert r.tl.x == pytest.approx(21.0, abs=0.2)
    assert r.br.x == pytest.approx(39.0, abs=0.2)
    assert r.tl.y == pytest.approx(18.6, abs=0.2)
    assert r.br.y == pytest.approx(25.0, abs=0.2)


def test_inject_synthesizes_family_rect_with_no_internal_keepout(tmp_path):
    # A keep-out-free footprint matched by a family glob -> exactly one injected
    # rect, no preserve. (This is the vendored-import case the inject path
    # exists for: the footprint dropped its keep-out, config restores it.)
    board = _board_with_footprint(
        str(tmp_path / "b.kicad_pcb"), _SOT23_PRETTY, _SOT23_NAME, ref="Q1"
    )
    spec = {"x_min": -5.0, "y_min": -8.0, "x_max": 5.0, "y_max": -3.0}
    rects = extract_keepout_rects(board, {"antenna_keepouts": {"*SOT-23-3*": spec}})
    assert len(rects) == 1 and rects[0].source == "inject", rects
    r = rects[0]
    assert r.owner_ref == "Q1"
    exp_tl, exp_br = _transform_local_rect(spec, 30.0, 35.0, 0.0)
    assert r.tl.x == pytest.approx(exp_tl.x) and r.tl.y == pytest.approx(exp_tl.y)
    assert r.br.x == pytest.approx(exp_br.x) and r.br.y == pytest.approx(exp_br.y)


def test_preserve_and_inject_union_for_matched_footprint(tmp_path):
    # WROOM has an internal strip (Fix 0) AND matches the family glob ->
    # both rects are emitted so strip + near-field are both honored.
    board = _board_with_wroom(str(tmp_path / "b.kicad_pcb"))
    rects = extract_keepout_rects(board, DEFAULT_CONFIG)
    sources = {r.source for r in rects}
    assert sources == {"preserve", "inject"}, rects
    assert all(r.owner_ref == "U1" for r in rects)


@pytest.mark.parametrize("rot", [0.0, 90.0, 180.0, 270.0])
def test_inject_transform_matches_pcbnew_placement(tmp_path, rot):
    """_transform_local_rect must reproduce pcbnew's actual placed geometry.

    Ground truth: the footprint's own internal keep-out, whose board-coord bbox
    pcbnew reports after placement (the preserve path). Feeding that zone's
    LOCAL rect through _transform_local_rect at the same pose must match.
    """
    # local rect = internal-zone bbox of the footprint at the origin
    fp0 = pcbnew.FootprintLoad(_S3_PRETTY, _NAME)
    z0 = list(fp0.Zones())[0].Outline().BBox()
    local = {
        "x_min": pcbnew.ToMM(z0.GetLeft()),
        "y_min": pcbnew.ToMM(z0.GetTop()),
        "x_max": pcbnew.ToMM(z0.GetRight()),
        "y_max": pcbnew.ToMM(z0.GetBottom()),
    }
    # pcbnew ground truth at pose (30, 35, rot) via the preserve path
    board = _board_with_wroom(str(tmp_path / "b.kicad_pcb"), rot=rot)
    truth = [r for r in extract_keepout_rects(board, {}) if r.source == "preserve"][0]
    got_tl, got_br = _transform_local_rect(local, 30.0, 35.0, rot)
    assert got_tl.x == pytest.approx(truth.tl.x, abs=0.05)
    assert got_tl.y == pytest.approx(truth.tl.y, abs=0.05)
    assert got_br.x == pytest.approx(truth.br.x, abs=0.05)
    assert got_br.y == pytest.approx(truth.br.y, abs=0.05)
