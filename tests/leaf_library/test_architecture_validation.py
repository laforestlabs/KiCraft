"""Tests for the architecture stage's library validation logic.

The LLM call itself is not exercised; we feed a hand-built Architecture
into ``_validate_library_picks`` to confirm error paths fire correctly.
"""

from __future__ import annotations

import pytest

from kicraft.circuitchat.models import (
    Architecture,
    InterSheetNet,
    Sheet,
    SheetPin,
)
from kicraft.circuitchat.library import (
    ArchitectureLibraryError,
    _format_available_leaves_block,
    _validate_library_picks,
)
from kicraft.leaf_library import LeafLibrary
from tests.leaf_library.test_loader import _populate_valid_leaf


def _make_loaded_leaves(tmp_path):
    _populate_valid_leaf(tmp_path / "test-leaf")
    lib = LeafLibrary(tmp_path)
    loaded, _ = lib.load_all()
    return loaded


def _make_arch(**kwargs) -> Architecture:
    base = dict(
        topologies={},
        rail_voltages={},
        comms_protocols=[],
        mcu_present=False,
        sheets=[],
        power_nets=["VBUS", "GND"],
        inter_sheet_nets=[],
        assumptions=[],
    )
    base.update(kwargs)
    return Architecture(**base)


def test_format_block_empty_returns_none():
    assert _format_available_leaves_block([]) is None


def test_format_block_includes_interface_and_metadata(tmp_path):
    leaves = _make_loaded_leaves(tmp_path)
    block = _format_available_leaves_block(leaves)
    assert block is not None
    assert "test-leaf@0.1.0" in block
    assert "VBUS (input)" in block


def test_validate_unknown_slug(tmp_path):
    leaves = _make_loaded_leaves(tmp_path)
    arch = _make_arch(
        sheets=[
            Sheet(
                name="CHARGER", stem="CHARGER", function="x",
                from_library="not-a-real@1.0.0", library_instance=1,
            ),
        ],
    )
    with pytest.raises(ArchitectureLibraryError, match="unknown library leaf"):
        _validate_library_picks(arch, leaves)


def test_validate_instance_gap(tmp_path):
    leaves = _make_loaded_leaves(tmp_path)
    # Two sheets, instance 1 and 3 (gap at 2).
    arch = _make_arch(
        sheets=[
            Sheet(
                name="CHARGER", stem="CHARGER", function="a",
                from_library="test-leaf@0.1.0", library_instance=1,
            ),
            Sheet(
                name="CHARGER B", stem="CHARGER_B", function="b",
                from_library="test-leaf@0.1.0", library_instance=3,
            ),
        ],
        inter_sheet_nets=[
            InterSheetNet(
                name="VBUS",
                endpoints=[
                    SheetPin(sheet="CHARGER", direction="input"),
                    SheetPin(sheet="CHARGER B", direction="input"),
                ],
            ),
        ],
    )
    with pytest.raises(ArchitectureLibraryError, match="must be sequential"):
        _validate_library_picks(arch, leaves)


def test_validate_interface_mismatch_extra(tmp_path):
    """Architecture has an extra net not in the leaf's interface."""
    leaves = _make_loaded_leaves(tmp_path)
    arch = _make_arch(
        sheets=[
            Sheet(
                name="CHARGER", stem="CHARGER", function="a",
                from_library="test-leaf@0.1.0", library_instance=1,
            ),
            Sheet(name="OTHER", stem="OTHER", function="o"),
        ],
        inter_sheet_nets=[
            InterSheetNet(
                name="VBUS",
                endpoints=[
                    SheetPin(sheet="CHARGER", direction="input"),
                    SheetPin(sheet="OTHER", direction="output"),
                ],
            ),
            InterSheetNet(
                name="EXTRA_NET",  # not in leaf interface
                endpoints=[
                    SheetPin(sheet="CHARGER", direction="output"),
                    SheetPin(sheet="OTHER", direction="input"),
                ],
            ),
        ],
    )
    with pytest.raises(ArchitectureLibraryError, match="interface mismatch"):
        _validate_library_picks(arch, leaves)


def test_validate_passes_when_interface_matches(tmp_path):
    leaves = _make_loaded_leaves(tmp_path)
    arch = _make_arch(
        sheets=[
            Sheet(
                name="CHARGER", stem="CHARGER", function="a",
                from_library="test-leaf@0.1.0", library_instance=1,
            ),
            Sheet(name="OTHER", stem="OTHER", function="o"),
        ],
        inter_sheet_nets=[
            InterSheetNet(
                name="VBUS",  # the only label in the leaf's interface
                endpoints=[
                    SheetPin(sheet="CHARGER", direction="input"),  # leaf's VBUS is input
                    SheetPin(sheet="OTHER", direction="output"),
                ],
            ),
        ],
    )
    # Must not raise.
    _validate_library_picks(arch, leaves)
