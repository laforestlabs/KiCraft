"""A leaf without internal nets must still produce a real, DRC-valid board."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from kicraft.autoplacer.brain import leaf_routing


class _CopyStampedBoard:
    """Keep fixture geometry fixed while exercising the real routing dispatch/DRC."""

    def __init__(self, source_pcb, config=None):
        self.source_pcb = source_pcb

    def stamp_subcircuit_board(self, board, *, output_path, **kwargs):
        shutil.copy2(self.source_pcb, output_path)


def _trivial_extraction(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        subcircuit=SimpleNamespace(
            schematic_path=str(tmp_path / "leaf.kicad_sch"), id="leaf-under-test"
        ),
        local_state=SimpleNamespace(
            nets={}, components={}, traces=[], vias=[], silkscreen=[], board_outline=None,
            board_width=20.0, board_height=20.0,
        ),
        internal_net_names=[],
        interface_ports=[],
        notes=[],
    )


@pytest.fixture
def fixed_leaf_geometry(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.setattr(
        leaf_routing, "resolve_artifact_paths",
        lambda root, sid: SimpleNamespace(artifact_dir=str(artifact_dir)),
    )
    monkeypatch.setattr(
        leaf_routing, "repair_leaf_placement_legality",
        lambda extraction, comps, cfg: (copy.deepcopy(comps), {"resolved": True}),
    )
    monkeypatch.setattr(leaf_routing, "KiCadAdapter", _CopyStampedBoard)
    monkeypatch.setattr(leaf_routing, "_outline_around_geometry", lambda comps, cfg: None)
    monkeypatch.setattr(leaf_routing, "_silk_for_leaf", lambda extraction, comps, cfg: [])
    return artifact_dir


def _prototyping_board(path: Path, *, overlapping_graphic: bool) -> None:
    pcbnew = pytest.importorskip("pcbnew")
    if shutil.which("kicad-cli") is None:
        pytest.skip("real KiCad DRC requires kicad-cli")
    # Build the fixture from the shipped footprint text: the fixed bundle is the
    # source of truth, and the legacy defect is the same file plus the unnetted
    # copper ring that shorted against the plated pad.
    shipped = (
        Path(__file__).resolve().parents[1] / "kicraft" / "parts_library"
        / "prototyping-area" / "prototyping-area.pretty"
        / "PrototypingPad_1.5mm_Drill0.8mm.kicad_mod"
    ).read_text()
    ring = (
        '  (fp_circle (center 0 0) (end 0.75 0) (stroke (width 0.1) (type default))'
        ' (fill none) (layer "F.Cu"))\n'
    )
    assert ring not in shipped, "the shipped pad must not carry the copper ring"
    library = path.parent / "proto.pretty"
    library.mkdir(exist_ok=True)
    (library / "PrototypingPad_1.5mm_Drill0.8mm.kicad_mod").write_text(
        shipped if not overlapping_graphic else shipped.replace("  (fp_rect", ring + "  (fp_rect", 1)
    )
    board = pcbnew.BOARD()
    footprint = pcbnew.FootprintLoad(str(library), "PrototypingPad_1.5mm_Drill0.8mm")
    assert footprint is not None
    footprint.SetReference("PB1")
    board.Add(footprint)
    footprint.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(10), pcbnew.FromMM(10)))
    corners = [(0, 0), (20, 0), (20, 20), (0, 20)]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(start[0]), pcbnew.FromMM(start[1])))
        edge.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(end[0]), pcbnew.FromMM(end[1])))
        edge.SetWidth(pcbnew.FromMM(0.05))
        board.Add(edge)
    pcbnew.SaveBoard(str(path), board)


@pytest.mark.parametrize("overlapping_graphic", [False, True])
def test_no_internal_nets_does_not_bypass_real_copper_drc(
    fixed_leaf_geometry, tmp_path, overlapping_graphic
):
    source = tmp_path / "seed.kicad_pcb"
    _prototyping_board(source, overlapping_graphic=overlapping_graphic)
    routing, _ = leaf_routing.route_local_subcircuit(
        _trivial_extraction(tmp_path), solved_components={},
        cfg={"pcb_path": str(source)}, generate_diagnostics=False, round_index=0,
    )
    assert routing["validation"]["drc"]["ran"] is True
    assert routing["failed"] is overlapping_graphic
    assert routing["validation"]["accepted"] is not overlapping_graphic
    if overlapping_graphic:
        assert routing["validation"]["drc"]["shorts"] > 0
    else:
        assert routing["validation"]["drc"]["shorts"] == 0
        assert Path(routing["routed_board_path"]).exists()


def test_missing_source_board_is_not_an_accepted_trivial_leaf(fixed_leaf_geometry, tmp_path):
    routing, _ = leaf_routing.route_local_subcircuit(
        _trivial_extraction(tmp_path), solved_components={},
        cfg={"pcb_path": str(tmp_path / "missing.kicad_pcb")}, generate_diagnostics=False,
    )
    assert routing["failed"] is True
    assert routing["validation"]["accepted"] is False
    assert routing["validation"]["board_exists"] is False
