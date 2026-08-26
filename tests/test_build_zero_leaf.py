"""`build` must fail fast with a clear message on a degenerate (0-leaf) hierarchy.

The "automatic dog feeder" web run reported `leafs=0/0` and then the build printed
the misleading "the layout engine produced no routed parent board ... board not
routable as placed". The real cause was a degenerate hierarchy: the root schematic
referenced no child sheets, so the hierarchical layout engine had nothing to
compose. These tests pin the early detection + the actionable message, so the user
is told to re-run synthesis instead of chasing a phantom routing problem."""
from __future__ import annotations

from pathlib import Path

import pytest

from kicraft.design import cli_app


@pytest.fixture
def schematics(tmp_path: Path) -> tuple[Path, Path]:
    flat = tmp_path / "flat.kicad_sch"
    flat.write_text("(kicad_sch (version 20231120))\n", encoding="utf-8")
    for name in ("child_a", "child_b"):
        (tmp_path / f"{name}.kicad_sch").write_text(
            "(kicad_sch (version 20231120))\n", encoding="utf-8"
        )
    proper = tmp_path / "proper.kicad_sch"
    proper.write_text(
        """(kicad_sch
  (version 20231120)
  (sheet
    (uuid "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    (property "Sheetname" "A")
    (property "Sheetfile" "child_a.kicad_sch"))
  (sheet
    (uuid "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    (property "Sheetname" "B")
    (property "Sheetfile" "child_b.kicad_sch")))
""",
        encoding="utf-8",
    )
    return flat, proper


def test_flat_schematic_has_zero_leaf_subcircuits(schematics):
    # A schematic with no child sheets is degenerate as a layout root: the engine's
    # non-root leaf count is 0 (exactly the run's `leafs=0/0`).
    flat, _proper = schematics
    assert cli_app._count_leaf_subcircuits(flat) == 0


def test_proper_hierarchy_has_leaf_subcircuits(schematics):
    _flat, proper = schematics
    assert cli_app._count_leaf_subcircuits(proper) >= 2


def test_degenerate_error_is_actionable_not_misleading(schematics):
    flat, proper = schematics
    msg = cli_app._degenerate_hierarchy_error(flat)
    assert msg is not None
    assert "no leaf subcircuits" in msg
    # the whole point: NOT the old misleading routing message.
    assert "not routable as placed" not in msg
    # a healthy hierarchy is not flagged.
    assert cli_app._degenerate_hierarchy_error(proper) is None
