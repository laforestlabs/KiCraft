"""`build` must fail fast with a clear message on a degenerate (0-leaf) hierarchy.

The "automatic dog feeder" web run reported `leafs=0/0` and then the build printed
the misleading "the layout engine produced no routed parent board ... board not
routable as placed". The real cause was a degenerate hierarchy: the root schematic
referenced no child sheets, so the hierarchical layout engine had nothing to
compose. These tests pin the early detection + the actionable message, so the user
is told to re-run synthesis instead of chasing a phantom routing problem."""
from __future__ import annotations

from pathlib import Path

from kicraft.design import cli_app

_FIXT = Path("tests/manual-runs/bmp280-reader/generated/USB_BMP280_READER")
_FLAT_ROOT = _FIXT / "USB_INPUT.kicad_sch"          # a leaf: no child sheets
_PROPER_ROOT = _FIXT / "USB_BMP280_READER.kicad_sch"  # root: 5 child leaves


def test_flat_schematic_has_zero_leaf_subcircuits():
    # A schematic with no child sheets is degenerate as a layout root: the engine's
    # non-root leaf count is 0 (exactly the run's `leafs=0/0`).
    assert cli_app._count_leaf_subcircuits(_FLAT_ROOT) == 0


def test_proper_hierarchy_has_leaf_subcircuits():
    assert cli_app._count_leaf_subcircuits(_PROPER_ROOT) >= 2


def test_degenerate_error_is_actionable_not_misleading():
    msg = cli_app._degenerate_hierarchy_error(_FLAT_ROOT)
    assert msg is not None
    assert "no leaf subcircuits" in msg
    # the whole point: NOT the old misleading routing message.
    assert "not routable as placed" not in msg
    # a healthy hierarchy is not flagged.
    assert cli_app._degenerate_hierarchy_error(_PROPER_ROOT) is None
