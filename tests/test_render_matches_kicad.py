"""Real saved-board render coordinates, with empty and displaced geometry controls."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

pcbnew = pytest.importorskip("pcbnew")
np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image

from kicraft.design.synthesis.footprint_library import FootprintNotFoundError, load_footprint
from kicraft.render import render_views

BOUNDS = (100.0, 75.0, 120.0, 95.0)
CENTERS = ((106.0, 81.0), (114.0, 89.0))


def _have_external_tools() -> bool:
    return shutil.which("kicad-cli") is not None and bool(
        shutil.which("magick") or shutil.which("convert")
        or importlib.util.find_spec("cairosvg")
    )


def _saved_board(path: Path, centers: tuple[tuple[float, float], ...]) -> None:
    board = pcbnew.NewBoard(str(path))
    x0, y0, x1, y1 = BOUNDS
    for start, end in (
        ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)),
    ):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(pcbnew.VECTOR2I(*(pcbnew.FromMM(v) for v in start)))
        edge.SetEnd(pcbnew.VECTOR2I(*(pcbnew.FromMM(v) for v in end)))
        edge.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge)
    for index, (x, y) in enumerate(centers, 1):
        try:
            footprint, _ = load_footprint(pcbnew, "TestPoint", "TestPoint_Pad_D1.5mm")
        except FootprintNotFoundError as exc:
            pytest.skip(f"stock test-point prerequisite unavailable: {exc}")
        footprint.SetReference(f"TP{index}")
        footprint.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y)))
        board.Add(footprint)
    assert board.Save(str(path)) is not False


def _native_centers(path: Path) -> list[tuple[str, float, float]]:
    board = pcbnew.LoadBoard(str(path))
    return [
        (footprint.GetReference(), pcbnew.ToMM(footprint.GetPosition().x),
         pcbnew.ToMM(footprint.GetPosition().y))
        for footprint in board.GetFootprints()
    ]


def _rgba_render(path: Path, output: Path):
    result = render_views(path, output, views=["front_all"])
    image = result.get("front_all")
    assert image is not None and Path(image).is_file()
    with Image.open(image) as opened:
        return np.asarray(opened.convert("RGBA"))


def _missing_geometry(image, empty, centers):
    assert image.shape == empty.shape
    # Compare full RGBA against a real empty-board render. Palette values and
    # transparent/substrate RGB cannot masquerade as component geometry.
    geometry = np.any(image != empty, axis=2)
    height, width = geometry.shape
    x0, y0, x1, y1 = BOUNDS
    missing = []
    for ref, x, y in centers:
        px = int(round((x - x0) / (x1 - x0) * width))
        py = int(round((y - y0) / (y1 - y0) * height))
        radius = 6
        patch = geometry[max(0, py - radius):min(height, py + radius + 1),
                         max(0, px - radius):min(width, px + radius + 1)]
        if not patch.any():
            missing.append(ref)
    return missing


@pytest.mark.skipif(not _have_external_tools(), reason="KiCad SVG export/rasterizer unavailable")
def test_gui_render_projects_footprints_to_correct_pixels(tmp_path):
    board, empty, displaced = (tmp_path / name for name in ("board.kicad_pcb", "empty.kicad_pcb", "displaced.kicad_pcb"))
    _saved_board(board, CENTERS)
    _saved_board(empty, ())
    _saved_board(displaced, ((106.0, 89.0), (114.0, 81.0)))
    centers = _native_centers(board)
    assert {ref for ref, _x, _y in centers} == {"TP1", "TP2"}
    background = _rgba_render(empty, tmp_path / "empty_render")
    actual = _rgba_render(board, tmp_path / "actual_render")
    shifted = _rgba_render(displaced, tmp_path / "displaced_render")
    assert _missing_geometry(actual, background, centers) == []
    assert set(_missing_geometry(shifted, background, centers)) == {"TP1", "TP2"}
