"""Unit tests for the R4 architecture inter-sheet contract checks.

Covers ``check_every_block_has_sheet`` and ``check_fs_connections_mapped`` in
``kicraft.design.synthesis.validation`` — the deterministic architecture-stage
gates that catch cross-sheet functional_spec connections never declared as
inter-sheet nets (the historical DTR/RTS->ESP32 and RESET/D0->PROTO defects)
and architectures that silently omit functional blocks.
"""

from kicraft.design.models import (
    Architecture,
    BlockConnection,
    CircuitRequirement,
    FabricationObligation,
    FunctionalBlock,
    FunctionalSpec,
    InterSheetNet,
    RecipeSelection,
    Sheet,
    SheetPin,
)
from kicraft.design.synthesis.validation import (
    check_every_block_has_sheet,
    check_fs_connections_mapped,
)


def _fs(*blocks, connections=None):
    return FunctionalSpec(blocks=list(blocks), connections=list(connections or []))


def _arch(sheets, inter_sheet_nets=None, requirements=(), obligations=()):
    return Architecture(
        sheets=list(sheets),
        power_nets=[],
        inter_sheet_nets=list(inter_sheet_nets or []),
        requirements=list(requirements),
        obligations=list(obligations),
    )


def _requirement(id, sheet, *blocks, ports=None):
    return CircuitRequirement(
        id=id,
        sheet=sheet,
        role="connector",
        family="header",
        functional_blocks=list(blocks),
        ports=ports or {},
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
        ],
        requirements=[
            _requirement("mcu", "MCU", "MCU"),
            _requirement("power", "POWER", "POWER"),
        ],
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
    assert any("MCU" in offender for offender in result.offenders)
    assert any("POWER" in offender for offender in result.offenders)


def test_board_feature_requirement_needs_no_functional_block():
    """A `fabrication` obligation derives a requirement no functional block can own.

    The proto-shield shape: the intent carries `fabrication`/`prototyping-area`, the compiler
    derives the PROTOTYPING AREA sheet and its pad-field requirement from that row, and the
    functional spec declares no block for the field at all -- a block is a user-visible function
    and bare pads carry no signal of their own. Requiring membership would refuse the board the
    brief asked for, at both R4 gates.
    """
    arch = _arch(
        [
            Sheet(name="POWER", stem="POWER", function="rail"),
            Sheet(name="PROTOTYPING AREA", stem="PROTOTYPING_AREA", function="pad field"),
        ],
        requirements=[
            _requirement("power", "POWER", "POWER"),
            CircuitRequirement(
                id="prototyping_area",
                sheet="PROTOTYPING AREA",
                role="user_io",
                family="prototyping-area",
                parameters={"rows": 5, "cols": 5, "pitch_mm": 2.54},
            ),
        ],
        obligations=[
            FabricationObligation(
                kind="fabrication",
                original_obligation_id="prototyping_area",
                feature="prototyping-area",
            )
        ],
    )
    fs = _fs(FunctionalBlock(name="POWER", category="power", purpose="rail"))

    assert check_every_block_has_sheet(fs, arch).ok is True
    assert check_fs_connections_mapped(fs, arch).ok is True


def test_only_the_board_feature_row_excuses_block_membership():
    """The exemption is keyed by the `fabrication` row, never by an empty block list.

    A requirement with no such row behind it keeps exactly the refusal it has today, so the
    gate cannot be cleared by simply leaving `functional_blocks` out -- and a `fabrication` row
    with another obligation id is a different fact that excuses nothing.
    """

    def offenders(obligation_ids: tuple[str, ...]) -> list[str]:
        return check_every_block_has_sheet(
            _fs(),
            _arch(
                [Sheet(name="PROTOTYPING AREA", stem="PROTOTYPING_AREA", function="pad field")],
                requirements=[
                    CircuitRequirement(
                        id="prototyping_area",
                        sheet="PROTOTYPING AREA",
                        role="user_io",
                        family="prototyping-area",
                    )
                ],
                obligations=[
                    FabricationObligation(
                        kind="fabrication",
                        original_obligation_id=source,
                        feature="prototyping-area",
                    )
                    for source in obligation_ids
                ],
            ),
        ).offenders

    missing = ["requirement 'prototyping_area' has no functional_blocks membership"]
    assert offenders(()) == missing
    assert offenders(("copper_pour_area",)) == missing
    assert offenders(("prototyping_area",)) == []


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
        requirements=[
            _requirement("mcu", "MCU", "MCU"),
            _requirement("programmer", "PROGRAMMER", "PROGRAMMER"),
        ],
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
        requirements=[
            _requirement("mcu", "MCU", "MCU", ports={"uart": "UART"}),
            _requirement("programmer", "PROGRAMMER", "PROGRAMMER", ports={"uart": "UART"}),
        ],
    )
    result = check_fs_connections_mapped(fs, arch)
    assert result.ok is True
    assert result.offenders == []


def test_fs_connections_mapped_shared_bus_covers_pairwise_connections():
    """A single inter_sheet_net declared across 3+ sheets (a shared bus, e.g.
    I2C over MCU/SENSOR/DISPLAY) covers every pairwise functional-spec
    connection between its endpoints. The gate used to require an EXACT
    2-sheet endpoint match, bouncing this correct architecture forever."""
    fs = _fs(
        FunctionalBlock(name="MCU", category="process", purpose="mcu"),
        FunctionalBlock(name="SENSOR", category="sense", purpose="sensor"),
        FunctionalBlock(name="DISPLAY", category="interface", purpose="display"),
        connections=[
            BlockConnection(
                from_block="MCU", to_block="SENSOR", signal_type="digital", description="I2C"
            ),
            BlockConnection(
                from_block="MCU", to_block="DISPLAY", signal_type="digital", description="I2C"
            ),
        ],
    )
    arch = _arch(
        [
            Sheet(name="MCU", stem="MCU", function="mcu"),
            Sheet(name="SENSOR", stem="SENSOR", function="sensor"),
            Sheet(name="DISPLAY", stem="DISPLAY", function="display"),
        ],
        inter_sheet_nets=[
            InterSheetNet(
                name="I2C_SDA",
                endpoints=[
                    SheetPin(sheet="MCU", direction="bidirectional"),
                    SheetPin(sheet="SENSOR", direction="bidirectional"),
                    SheetPin(sheet="DISPLAY", direction="bidirectional"),
                ],
            ),
        ],
        requirements=[
            _requirement("mcu", "MCU", "MCU", ports={"sda": "I2C_SDA"}),
            _requirement("sensor", "SENSOR", "SENSOR", ports={"sda": "I2C_SDA"}),
            _requirement("display", "DISPLAY", "DISPLAY", ports={"sda": "I2C_SDA"}),
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
        requirements=[
            _requirement("mcu", "MCU", "MCU"),
            _requirement("power", "POWER", "POWER"),
        ],
    )
    result = check_fs_connections_mapped(fs, arch)
    assert result.ok is True
    assert result.offenders == []


def test_merged_usb_and_header_requirements_cover_both_functions():
    fs = _fs(
        FunctionalBlock(name="USB_C_RECEPTACLE", category="interface", purpose="USB receptacle"),
        FunctionalBlock(name="BREAKOUT_HEADER", category="interface", purpose="Expose all signals"),
        connections=[
            BlockConnection(
                from_block="USB_C_RECEPTACLE", to_block="BREAKOUT_HEADER", signal_type="digital"
            )
        ],
    )
    arch = _arch(
        [Sheet(name="MAIN", stem="MAIN", function="USB receptacle and breakout header")],
        requirements=[
            CircuitRequirement(
                id="usb",
                sheet="MAIN",
                role="connector",
                family="usb-c-breakout",
                functional_blocks=["USB_C_RECEPTACLE"],
                ports={"vbus": "VBUS", "gnd": "GND", "cc1": "CC1", "sbu1": "SBU1"},
            ),
            _requirement("header", "MAIN", "BREAKOUT_HEADER"),
        ],
    )
    assert check_every_block_has_sheet(fs, arch).ok
    assert check_fs_connections_mapped(fs, arch).ok

    # The recipe-populated merged sheet cannot hide the omitted header.
    arch.requirements.pop()
    assert not check_every_block_has_sheet(fs, arch).ok
    assert not check_fs_connections_mapped(fs, arch).ok


def test_sheet_names_and_prose_do_not_create_block_membership():
    fs = _fs(FunctionalBlock(name="PD_TRIGGER", category="process", purpose="PD negotiation"))
    arch = _arch([Sheet(name="PD TRIGGER", stem="PD_TRIGGER", function="PD negotiation")])
    assert not check_every_block_has_sheet(fs, arch).ok
    arch.requirements = [_requirement("pd", "PD TRIGGER")]
    assert not check_every_block_has_sheet(fs, arch).ok
    arch.requirements[0].functional_blocks = ["PD_TRIGGER"]
    assert check_every_block_has_sheet(fs, arch).ok
    arch.requirements.append(_requirement("unknown", "PD TRIGGER", "PD"))
    assert not check_every_block_has_sheet(fs, arch).ok


def test_composite_and_multiple_owners_use_explicit_cross_sheet_relation():
    fs = _fs(
        FunctionalBlock(name="CONTROL", category="process", purpose="Control"),
        FunctionalBlock(name="POWER", category="power", purpose="Power"),
        FunctionalBlock(name="IO", category="interface", purpose="User IO"),
        connections=[BlockConnection(from_block="CONTROL", to_block="IO", signal_type="digital")],
    )
    arch = _arch(
        [
            Sheet(name="CORE", stem="CORE", function="Composite circuit"),
            Sheet(name="AUX", stem="AUX", function="Additional control"),
            Sheet(name="PORTS", stem="PORTS", function="User connections"),
        ],
        requirements=[
            _requirement("composite", "CORE", "CONTROL", "POWER"),
            _requirement("aux", "AUX", "CONTROL", ports={"tx": "DATA"}),
            _requirement("io", "PORTS", "IO", ports={"rx": "DATA"}),
        ],
        inter_sheet_nets=[
            InterSheetNet(
                name="DATA",
                endpoints=[
                    SheetPin(sheet="AUX", direction="output"),
                    SheetPin(sheet="PORTS", direction="input"),
                ],
            )
        ],
    )
    assert check_every_block_has_sheet(fs, arch).ok
    assert check_fs_connections_mapped(fs, arch).ok
    arch.inter_sheet_nets.clear()
    assert not check_fs_connections_mapped(fs, arch).ok


def test_connection_gate_rejects_unknown_or_unmapped_power_endpoints():
    fs = _fs(
        FunctionalBlock(name="SOURCE", category="power", purpose="Source"),
        FunctionalBlock(name="LOAD", category="process", purpose="Load"),
        connections=[BlockConnection(from_block="SOURCE", to_block="LOAD", signal_type="power")],
    )
    arch = _arch(
        [Sheet(name="MAIN", stem="MAIN", function="Power and load")],
        requirements=[_requirement("source", "MAIN", "SOURCE")],
    )
    assert not check_fs_connections_mapped(fs, arch).ok
    arch.requirements.append(_requirement("load", "MAIN", "LOAD"))
    assert check_fs_connections_mapped(fs, arch).ok
    # Protect callers that receive a constructed/modified model, too.
    fs.connections[0].to_block = "UNKNOWN"
    assert not check_fs_connections_mapped(fs, arch).ok


def test_recipe_resolved_usb_sink_does_not_cover_breakout_functional_spec():
    # Reduced gate inputs from round3/round4: one populated sheet, one
    # inferred/resolved sink requirement, no explicit ownership for either block.
    fs = _fs(
        FunctionalBlock(name="USB_C_RECEPTACLE", category="interface", purpose="USB connector"),
        FunctionalBlock(name="HEADER_BREAKOUT", category="interface", purpose="Breakout headers"),
        connections=[
            BlockConnection(
                from_block="USB_C_RECEPTACLE", to_block="HEADER_BREAKOUT", signal_type="bus"
            )
        ],
    )
    arch = Architecture.model_validate(
        {
            "sheets": [
                {
                    "name": "USB C BREAKOUT",
                    "stem": "USB_C_BREAKOUT",
                    "function": "USB-C receptacle and breakout headers",
                }
            ],
            "power_nets": ["VBUS", "GND"],
            "inter_sheet_nets": [],
            "requirements": [
                {
                    "id": "auto_usb_c_usb_c_breakout",
                    "sheet": "USB C BREAKOUT",
                    "role": "power_input",
                    "family": "usb-c-power-sink",
                    "exact_part": "USB-C-5V-SINK",
                    "ports": {"gnd": "GND", "vbus": "VBUS"},
                }
            ],
            "recipe_selections": [
                {
                    "recipe": "usb-c-5v-sink@1",
                    "instance": "auto_usb_c_usb_c_breakout",
                    "sheets": {"power": "USB C BREAKOUT"},
                    "requirement_ids": ["auto_usb_c_usb_c_breakout"],
                }
            ],
        }
    )
    assert not check_every_block_has_sheet(fs, arch).ok
    assert not check_fs_connections_mapped(fs, arch).ok


def test_empty_pd_architecture_does_not_cover_committed_functions():
    # Reduced round4 gate inputs, not a modified historical live state.
    fs = _fs(
        FunctionalBlock(name="USB_C_INPUT", category="interface", purpose="USB input"),
        FunctionalBlock(name="PD_NEGOTIATION", category="process", purpose="PD controller"),
        FunctionalBlock(name="VOLTAGE_SELECT", category="sense", purpose="Selection switch"),
        FunctionalBlock(name="OUTPUT_POWER", category="power", purpose="Load connection"),
        connections=[
            BlockConnection(
                from_block="USB_C_INPUT", to_block="PD_NEGOTIATION", signal_type="digital"
            ),
            BlockConnection(from_block="USB_C_INPUT", to_block="OUTPUT_POWER", signal_type="power"),
        ],
    )
    arch = _arch(
        [
            Sheet(
                name="PD TRIGGER", stem="PD_TRIGGER", function="USB-C PD negotiation and power path"
            )
        ]
    )
    assert not check_every_block_has_sheet(fs, arch).ok
    assert not check_fs_connections_mapped(fs, arch).ok


def test_usb_declared_endpoints_require_net_values_on_recipe_and_model_owned_sheets():
    fs = _fs(
        FunctionalBlock(name="USB", category="interface", purpose="USB receptacle"),
        FunctionalBlock(name="HEADER", category="interface", purpose="Breakout header"),
        connections=[BlockConnection(from_block="USB", to_block="HEADER", signal_type="bus")],
    )
    net_names = ("VBUS", "GND", "CC1", "CC2", "SBU1", "SBU2", "D_P", "D_N")
    directions = {
        name.lower(): "power" if name == "VBUS" else "ground" if name == "GND" else "bidirectional"
        for name in net_names
    }
    arch = _arch(
        [
            Sheet(name="USB", stem="USB", function="USB connector"),
            Sheet(name="HEADER", stem="HEADER", function="Breakout header"),
        ],
        requirements=[
            CircuitRequirement(
                id="usb",
                sheet="USB",
                role="power_input",
                family="usb-c-power-sink",
                functional_blocks=["USB"],
                ports=directions,
            ),
            _requirement("header", "HEADER", "HEADER", ports=directions),
        ],
        inter_sheet_nets=[
            InterSheetNet(
                name=name,
                endpoints=[
                    SheetPin(sheet=sheet, direction="bidirectional") for sheet in ("USB", "HEADER")
                ],
            )
            for name in net_names
        ],
    )
    arch.recipe_selections = [
        RecipeSelection(
            recipe="usb-c-5v-sink@1",
            instance="usb",
            sheets={"power": "USB"},
            requirement_ids=["usb"],
            port_bindings={"gnd": "GND", "vbus": "VBUS"},
        )
    ]
    # Both functions are owned, but Round5's direction-valued bindings wire none
    # of the declared nets. Power and ground declarations are contracts, too.
    assert check_every_block_has_sheet(fs, arch).ok
    rejected = check_fs_connections_mapped(fs, arch)
    assert not rejected.ok
    for name in net_names:
        for sheet in ("USB", "HEADER"):
            assert any(
                repr(name) in item and f"sheet {sheet!r}" in item for item in rejected.offenders
            )

    arch.requirements[0].ports = {name.lower(): name for name in net_names}
    rejected = check_fs_connections_mapped(fs, arch)
    assert not rejected.ok
    assert all("sheet 'HEADER'" in item for item in rejected.offenders)

    # Multiple model-owned requirements may jointly implement the same sheet.
    arch.requirements[1].ports = {name.lower(): name for name in net_names[:4]}
    arch.requirements.append(
        _requirement(
            "header_signals",
            "HEADER",
            "HEADER",
            ports={name.lower(): name for name in net_names[4:]},
        )
    )
    assert check_fs_connections_mapped(fs, arch).ok
    arch.requirements[2].ports["d_p"] = "d_p"
    assert not check_fs_connections_mapped(fs, arch).ok


def test_direction_words_are_valid_net_names_when_explicitly_bound():
    fs = _fs(FunctionalBlock(name="IO", category="interface", purpose="IO"))
    arch = _arch(
        [
            Sheet(name="A", stem="A", function="Source"),
            Sheet(name="B", stem="B", function="Receiver"),
        ],
        requirements=[
            _requirement("a", "A", "IO", ports={"tx": "input", "local": "output"}),
            _requirement("b", "B", "IO", ports={"rx": "input"}),
        ],
        inter_sheet_nets=[
            InterSheetNet(
                name="input",
                endpoints=[
                    SheetPin(sheet="A", direction="output"),
                    SheetPin(sheet="B", direction="input"),
                ],
            )
        ],
    )
    assert check_fs_connections_mapped(fs, arch).ok
