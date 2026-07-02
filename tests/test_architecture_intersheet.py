"""Unit tests for the R4 architecture inter-sheet contract checks.

Covers ``check_every_block_has_sheet`` and ``check_fs_connections_mapped`` in
``kicraft.design.synthesis.validation`` — the deterministic architecture-stage
gates that catch cross-sheet functional_spec connections never declared as
inter-sheet nets (the historical DTR/RTS->ESP32 and RESET/D0->PROTO defects)
and architectures with zero sheets despite blocks.
"""

from kicraft.design.models import (
    Architecture,
    BlockConnection,
    FunctionalBlock,
    FunctionalSpec,
    InterSheetNet,
    Sheet,
    SheetPin,
)
from kicraft.design.synthesis.validation import (
    check_every_block_has_sheet,
    check_fs_connections_mapped,
)


def _fs(*blocks, connections=None):
    return FunctionalSpec(blocks=list(blocks), connections=list(connections or []))


def _arch(sheets, inter_sheet_nets=None):
    return Architecture(
        sheets=list(sheets),
        power_nets=[],
        inter_sheet_nets=list(inter_sheet_nets or []),
    )


# ---------------------------------------------------------------------------
# check_every_block_has_sheet
# ---------------------------------------------------------------------------


def test_block_has_sheet_ok():
    fs = _fs(
        FunctionalBlock(name="MCU", category="process", purpose="mcu"),
        FunctionalBlock(name="POWER", category="power", purpose="rail"),
    )
    arch = _arch(
        [
            Sheet(name="MCU", stem="MCU", function="mcu"),
            Sheet(name="POWER", stem="POWER", function="rail"),
        ]
    )
    result = check_every_block_has_sheet(fs, arch)
    assert result.ok is True
    assert result.offenders == []


def test_block_has_sheet_zero_sheets():
    fs = _fs(
        FunctionalBlock(name="MCU", category="process", purpose="mcu"),
        FunctionalBlock(name="POWER", category="power", purpose="rail"),
    )
    arch = _arch([])
    result = check_every_block_has_sheet(fs, arch)
    assert result.ok is False
    assert result.offenders, "expected an offender for zero sheets"
    assert any("zero sheets" in o for o in result.offenders)


# ---------------------------------------------------------------------------
# check_fs_connections_mapped
# ---------------------------------------------------------------------------


def test_fs_connections_mapped_cross_sheet_unmapped():
    fs = _fs(
        FunctionalBlock(name="MCU", category="process", purpose="mcu"),
        FunctionalBlock(name="PROGRAMMER", category="interface", purpose="prog"),
        connections=[
            BlockConnection(
                from_block="MCU",
                to_block="PROGRAMMER",
                signal_type="digital",
                description="UART",
            ),
        ],
    )
    arch = _arch(
        [
            Sheet(name="MCU", stem="MCU", function="mcu"),
            Sheet(name="PROGRAMMER", stem="PROGRAMMER", function="prog"),
        ],
        # No inter_sheet_net declares the MCU<->PROGRAMMER crossing.
        inter_sheet_nets=[],
    )
    result = check_fs_connections_mapped(fs, arch)
    assert result.ok is False
    joined = " ".join(result.offenders)
    assert "MCU" in joined
    assert "PROGRAMMER" in joined


def test_fs_connections_mapped_cross_sheet_covered():
    fs = _fs(
        FunctionalBlock(name="MCU", category="process", purpose="mcu"),
        FunctionalBlock(name="PROGRAMMER", category="interface", purpose="prog"),
        connections=[
            BlockConnection(
                from_block="MCU",
                to_block="PROGRAMMER",
                signal_type="digital",
                description="UART",
            ),
        ],
    )
    arch = _arch(
        [
            Sheet(name="MCU", stem="MCU", function="mcu"),
            Sheet(name="PROGRAMMER", stem="PROGRAMMER", function="prog"),
        ],
        inter_sheet_nets=[
            InterSheetNet(
                name="UART",
                endpoints=[
                    SheetPin(sheet="MCU", direction="bidirectional"),
                    SheetPin(sheet="PROGRAMMER", direction="bidirectional"),
                ],
            ),
        ],
    )
    result = check_fs_connections_mapped(fs, arch)
    assert result.ok is True
    assert result.offenders == []


def test_fs_connections_mapped_power_exempt():
    fs = _fs(
        FunctionalBlock(name="MCU", category="process", purpose="mcu"),
        FunctionalBlock(name="POWER", category="power", purpose="rail"),
        connections=[
            BlockConnection(
                from_block="POWER",
                to_block="MCU",
                signal_type="power",
                description="+3V3 rail",
            ),
        ],
    )
    arch = _arch(
        [
            Sheet(name="MCU", stem="MCU", function="mcu"),
            Sheet(name="POWER", stem="POWER", function="rail"),
        ],
        # No inter_sheet_net — power/ground are exempt (global power symbols).
        inter_sheet_nets=[],
    )
    result = check_fs_connections_mapped(fs, arch)
    assert result.ok is True
    assert result.offenders == []
