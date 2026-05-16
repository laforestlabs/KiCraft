"""Unit tests for upstream Pydantic models.

Locks the validation contract:
- reference designator regex
- footprint / symbol Library:Name shape
- sheet name and stem patterns
- power/ground net name recognition (mirrors §2.5)
- BOM cross-references (ic_groups, thermal_refs, signal_flow_order, component_zones)
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kicraft.upstream.models import (
    Architecture,
    BOM,
    BomPart,
    ConversationState,
    FunctionalBlock,
    FunctionalSpec,
    IntentSlot,
    InterSheetNet,
    Question,
    Sheet,
    SheetPin,
    is_power_or_ground_name,
)


# ---------- Power/ground name recognition (mirrors contract §2.5) ----------


@pytest.mark.parametrize(
    "name",
    ["VCC", "VDD", "VBAT", "VBUS", "VSYS", "+5V", "+3V3", "+3.3V", "3V3", "3.3V", "5V", "+12V"],
)
def test_power_names_recognized(name: str) -> None:
    assert is_power_or_ground_name(name)


@pytest.mark.parametrize("name", ["GND", "PGND", "AGND", "DGND", "VBAT_GND"])
def test_ground_names_recognized(name: str) -> None:
    assert is_power_or_ground_name(name)


@pytest.mark.parametrize("name", ["MOSI", "SCL", "CHG_STATUS", "BATT_POSITIVE", "EARTH", "DATA0"])
def test_signal_names_not_recognized_as_power(name: str) -> None:
    assert not is_power_or_ground_name(name)


def test_slashed_variants_recognized() -> None:
    # Child-sheet local namespace prefixes — §4.2 of the contract.
    assert is_power_or_ground_name("/VBAT")
    assert is_power_or_ground_name("/+3V3")


# ---------- Reference designator regex (§2.4) ----------


@pytest.mark.parametrize("ref", ["U1", "C12", "R3", "RT1", "H86", "BT1", "LED7", "H4_GND"])
def test_valid_refs(ref: str) -> None:
    p = BomPart(ref=ref, value="x", symbol="L:N", footprint="L:N", sheet="USB INPUT")
    assert p.ref == ref


@pytest.mark.parametrize("ref", ["1U", "U", "u1", "U-1", "MyPart", "U.1", "U/1", "_U1"])
def test_invalid_refs_rejected(ref: str) -> None:
    with pytest.raises(ValidationError):
        BomPart(ref=ref, value="x", symbol="L:N", footprint="L:N", sheet="USB INPUT")


# ---------- Footprint Library:Name shape (§2.4) ----------


def test_empty_footprint_rejected() -> None:
    with pytest.raises(ValidationError):
        BomPart(ref="U1", value="x", symbol="L:N", footprint="", sheet="S")


def test_unscoped_footprint_rejected() -> None:
    with pytest.raises(ValidationError):
        BomPart(ref="U1", value="x", symbol="L:N", footprint="SOT-23", sheet="S")


def test_valid_footprint() -> None:
    p = BomPart(
        ref="U1",
        value="AP2112K-3.3",
        symbol="Regulator_Linear:AP2112K-3.3",
        footprint="Package_TO_SOT_SMD:SOT-23-5",
        sheet="LDO 3V3",
    )
    assert p.footprint == "Package_TO_SOT_SMD:SOT-23-5"


# ---------- Sheet naming (§7) ----------


def test_sheet_name_must_be_uppercase() -> None:
    with pytest.raises(ValidationError):
        Sheet(name="Usb Input", stem="USB_INPUT", function="x")


def test_sheet_stem_no_spaces() -> None:
    with pytest.raises(ValidationError):
        Sheet(name="USB INPUT", stem="USB INPUT", function="x")


def test_sheet_ok() -> None:
    s = Sheet(name="BOOST 5V", stem="BOOST_5V", function="5V boost converter")
    assert s.stem == "BOOST_5V"


# ---------- Architecture cross-refs ----------


def _two_sheet_arch() -> Architecture:
    return Architecture(
        sheets=[
            Sheet(name="INPUT", stem="INPUT", function="usb in"),
            Sheet(name="REG", stem="REG", function="ldo"),
        ],
        power_nets=["VBUS", "+3V3", "GND"],
        inter_sheet_nets=[
            InterSheetNet(
                name="VBUS",
                endpoints=[
                    SheetPin(sheet="INPUT", direction="output"),
                    SheetPin(sheet="REG", direction="input"),
                ],
            )
        ],
    )


def test_architecture_valid() -> None:
    _two_sheet_arch()


def test_inter_sheet_net_unknown_sheet_rejected() -> None:
    with pytest.raises(ValidationError):
        Architecture(
            sheets=[Sheet(name="INPUT", stem="INPUT", function="x")],
            power_nets=["VBUS"],
            inter_sheet_nets=[
                InterSheetNet(
                    name="VBUS",
                    endpoints=[
                        SheetPin(sheet="INPUT", direction="output"),
                        SheetPin(sheet="GHOST", direction="input"),
                    ],
                )
            ],
        )


def test_duplicate_sheet_names_rejected() -> None:
    with pytest.raises(ValidationError):
        Architecture(
            sheets=[
                Sheet(name="INPUT", stem="INPUT", function="x"),
                Sheet(name="INPUT", stem="INPUT2", function="y"),
            ],
            power_nets=["VBUS"],
            inter_sheet_nets=[],
        )


def test_inter_sheet_net_needs_two_endpoints() -> None:
    with pytest.raises(ValidationError):
        InterSheetNet(name="VBUS", endpoints=[SheetPin(sheet="A", direction="output")])


# ---------- BOM cross-refs ----------


def _bom_with_groups() -> BOM:
    return BOM(
        parts=[
            BomPart(
                ref="U1",
                value="LDO",
                symbol="Regulator_Linear:AP2112K-3.3",
                footprint="Package_TO_SOT_SMD:SOT-23-5",
                sheet="REG",
            ),
            BomPart(
                ref="C1",
                value="1uF",
                symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric",
                sheet="REG",
            ),
            BomPart(
                ref="C2",
                value="1uF",
                symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric",
                sheet="REG",
            ),
        ],
        ic_groups={"U1": ["C1", "C2"]},
        group_labels={"U1": "REGULATOR"},
        thermal_refs=["U1"],
        signal_flow_order=["U1"],
    )


def test_bom_valid() -> None:
    bom = _bom_with_groups()
    assert len(bom.parts) == 3


def test_bom_duplicate_refs_rejected() -> None:
    with pytest.raises(ValidationError):
        BOM(
            parts=[
                BomPart(ref="U1", value="x", symbol="L:N", footprint="L:N", sheet="S"),
                BomPart(ref="U1", value="y", symbol="L:N", footprint="L:N", sheet="S"),
            ]
        )


def test_bom_unknown_ic_group_member_rejected() -> None:
    with pytest.raises(ValidationError):
        BOM(
            parts=[
                BomPart(ref="U1", value="x", symbol="L:N", footprint="L:N", sheet="S"),
            ],
            ic_groups={"U1": ["C99"]},
        )


def test_bom_unknown_thermal_ref_rejected() -> None:
    with pytest.raises(ValidationError):
        BOM(
            parts=[
                BomPart(ref="U1", value="x", symbol="L:N", footprint="L:N", sheet="S"),
            ],
            thermal_refs=["U999"],
        )


# ---------- ConversationState slot-scoped question replacement ----------


def test_replace_open_questions_for_stage() -> None:
    s = ConversationState(
        open_questions=[
            Question(text="a?", stage="intent"),
            Question(text="b?", stage="functional_spec"),
        ]
    )
    s.replace_open_questions_for_stage("intent", [Question(text="a2?", stage="intent")])
    by_stage = {q.stage: q.text for q in s.open_questions}
    assert by_stage == {"intent": "a2?", "functional_spec": "b?"}


def test_intent_assumptions_recorded() -> None:
    slot = IntentSlot(goal="x", assumptions=["package: SMD (defaulted)"])
    assert "package: SMD (defaulted)" in slot.assumptions


def test_functional_spec_unknown_block_rejected() -> None:
    with pytest.raises(ValidationError):
        FunctionalSpec(
            blocks=[FunctionalBlock(name="A", category="process", purpose="x")],
            connections=[
                {
                    "from_block": "A",
                    "to_block": "GHOST",
                    "signal_type": "digital",
                }
            ],
        )
