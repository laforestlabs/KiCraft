"""A brief that asks for a four-layer stack-up must actually get one.

Before this, the request reached no gate: the intent kept it as prose in ``constraints``, the
builder emitted two copper layers, and the fab package carried two copper gerbers. These tests
pin each seam -- recording the request, declaring it on the seed board the whole pipeline
inherits, refusing a count no stack-up can carry, plotting the layers the board declares, and
keeping the router off the inner planes (KiCraft's copper bookkeeping carries front/back only,
so a signal routed on an inner layer would come back as front copper and be rebuilt there).
"""
from __future__ import annotations

import pytest

from kicraft.design.stage_semantics import (
    complete_intent_classification,
    requested_board_copper_layers,
)
from kicraft.design.synthesis.fab_export import fab_stack_layers

FOUR_LAYER_BRIEF = (
    "A 5 V header from the host board to 3.3 V converter with reverse-polarity protection, a "
    "6-pin 0.1 inch header, and a power LED. Use a four-layer stack-up."
)


def test_stated_stack_up_is_recorded_as_a_board_obligation() -> None:
    completed = complete_intent_classification(FOUR_LAYER_BRIEF, {"obligations": []})
    row = completed["obligations"][-1]
    assert row["kind"] == "quantitative"
    assert row["quantity"] == "PCB copper layers"
    assert row["value"] == 4.0
    assert row["unit"] == "layers"


@pytest.mark.parametrize(
    ("brief", "expected"),
    [
        (FOUR_LAYER_BRIEF, 4),
        ("Use a two-layer stack-up.", 2),
        ("A 6-layer PCB with a radio.", 6),
        # A count that merely sits near a number is not a stack-up, and a count no stack-up can
        # carry is left unrecorded rather than turned into a failed build.
        ("Keep it under 100 x 100 mm and use a power LED.", None),
        ("Make it a hundred-layer board.", None),
    ],
)
def test_the_recorded_count_comes_from_the_briefs_own_words(brief, expected) -> None:
    assert requested_board_copper_layers(brief) == expected
    row = complete_intent_classification(brief, {"obligations": []})["obligations"]
    recorded = [item for item in row if item.get("quantity") == "PCB copper layers"]
    assert len(recorded) == (0 if expected is None else 1)


def test_a_writers_own_stack_up_row_is_left_alone() -> None:
    written = {
        "kind": "quantitative",
        "original_obligation_id": "pcb_layers",
        "quantity": "PCB copper layers",
        "relation": "equal",
        "value": 2.0,
        "unit": "layers",
    }
    completed = complete_intent_classification(FOUR_LAYER_BRIEF, {"obligations": [written]})
    assert completed["obligations"] == [written]
    assert complete_intent_classification(FOUR_LAYER_BRIEF, completed) == completed


def _board(tmp_path, name: str, copper_layers: int):
    pcbnew = pytest.importorskip("pcbnew")
    path = tmp_path / name
    board = pcbnew.NewBoard(str(path))
    board.SetCopperLayerCount(copper_layers)
    board.Save(str(path))
    return path


def test_the_seed_board_declares_the_requested_layers(tmp_path) -> None:
    from kicraft.design.synthesis.kicad_pcb_stub import write_empty_pcb

    pcbnew = pytest.importorskip("pcbnew")
    out = write_empty_pcb(tmp_path, "FOURUP", None, copper_layers=4)
    board = pcbnew.LoadBoard(str(out))
    assert board.GetCopperLayerCount() == 4
    layers = [board.GetLayerName(layer) for layer in board.GetEnabledLayers().CuStack()]
    assert layers == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]

    # The default board every earlier design gets is untouched.
    default = pcbnew.LoadBoard(str(write_empty_pcb(tmp_path, "TWOUP", None)))
    assert default.GetCopperLayerCount() == 2


def test_the_fab_stack_follows_the_board_it_plots(tmp_path) -> None:
    from kicraft.design.synthesis.fab_export import _FAB_LAYERS

    four = fab_stack_layers(str(_board(tmp_path, "four.kicad_pcb", 4)))
    assert four.split(",")[:4] == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    assert fab_stack_layers(str(_board(tmp_path, "two.kicad_pcb", 2))) == _FAB_LAYERS


def test_the_router_keeps_signals_off_the_inner_planes(tmp_path) -> None:
    from kicraft.autoplacer import kicad_routing_tools as krt

    four = _board(tmp_path, "four.kicad_pcb", 4)
    config = {"kicad_routing_tools_path": str(tmp_path / "krt")}
    cmd = krt._krt_command(str(four), str(tmp_path / "out.kicad_pcb"), config, tmp_path / "fab.json")
    assert cmd[cmd.index("--layers") + 1:cmd.index("--layers") + 3] == ["F.Cu", "B.Cu"]

    two = _board(tmp_path, "two.kicad_pcb", 2)
    assert "--layers" not in krt._krt_command(
        str(two), str(tmp_path / "out.kicad_pcb"), config, tmp_path / "fab.json"
    )

    # An explicit routing-layer config still wins over the outer-pair default.
    pinned = krt._krt_command(
        str(four), str(tmp_path / "out.kicad_pcb"),
        {**config, "kicad_routing_tools_layers": ["F.Cu"]}, tmp_path / "fab.json",
    )
    assert pinned[pinned.index("--layers") + 1] == "F.Cu"


def test_the_gnd_plane_grows_to_the_inner_layers() -> None:
    from kicraft.autoplacer.brain.gnd_pour import _plane_layers

    four = {"F.Cu": 0, "In1.Cu": 4, "In2.Cu": 6, "B.Cu": 2}
    assert _plane_layers(four, ("B.Cu",)) == ("B.Cu", "In1.Cu", "In2.Cu")
    assert _plane_layers(four, ("B.Cu", "F.Cu")) == ("B.Cu", "F.Cu", "In1.Cu", "In2.Cu")
    assert _plane_layers({"F.Cu": 0, "B.Cu": 2}, ("B.Cu", "F.Cu")) == ("B.Cu", "F.Cu")
