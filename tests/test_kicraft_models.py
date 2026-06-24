"""Unit tests for KiCraft Pydantic models.

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

from kicraft.design.models import (
    Architecture,
    ArraySpec,
    BOM,
    BomPart,
    ConversationState,
    FunctionalBlock,
    FunctionalSpec,
    IntentSlot,
    InterSheetNet,
    NetConnection,
    PinEndpoint,
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


@pytest.mark.parametrize(
    "name",
    ["-12V", "-5V", "-3.3V", "-3V3", "VEE", "VSS", "/-12V", "/VEE"],
)
def test_negative_supply_rails_recognized(name: str) -> None:
    # Dual-supply / negative-rail boards (op-amp front-ends, audio): without
    # this the negative rail gets no PWR_FLAG and ERC flags VCC- as undriven.
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


def test_functional_block_count_defaults_one_and_must_be_positive() -> None:
    assert FunctionalBlock(name="A", category="drive", purpose="x").count == 1
    b = FunctionalBlock(name="STEPPER", category="drive", purpose="x", count=3)
    assert b.count == 3
    with pytest.raises(ValidationError):
        FunctionalBlock(name="A", category="drive", purpose="x", count=0)


def test_sheet_replication_fields_ok_and_paired() -> None:
    s = Sheet(
        name="STEPPER AXIS X", stem="STEPPER_AXIS_X", function="x",
        replication_group="STEPPER_AXIS", replication_instance=1,
    )
    assert s.replication_group == "STEPPER_AXIS" and s.replication_instance == 1
    # unpaired -> rejected
    with pytest.raises(ValidationError):
        Sheet(name="X", stem="X", function="x", replication_group="G")
    with pytest.raises(ValidationError):
        Sheet(name="X", stem="X", function="x", replication_instance=2)
    # instance must be >= 1
    with pytest.raises(ValidationError):
        Sheet(name="X", stem="X", function="x",
              replication_group="G", replication_instance=0)
    # cannot be both library reuse and a replication instance
    with pytest.raises(ValidationError):
        Sheet(name="X", stem="X", function="x", from_library="leaf@1",
              library_instance=1, replication_group="G", replication_instance=1)


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


# ---------- PinEndpoint / NetConnection ----------


def test_pin_endpoint_valid() -> None:
    ep = PinEndpoint(ref="U1", pin="3")
    assert ep.ref == "U1"
    assert ep.pin == "3"


# "1'"/"2'" are prime-notation pins from real symbols (e.g. the EVQ-P7A01P
# tactile switch, whose paired terminals are 1/1' and 2/2'); the wiring stage
# must be able to reference them.
@pytest.mark.parametrize(
    "pin", ["3", "A1", "B12", "VBAT", "+3V3", "~RESET", "GPIO0", "1'", "2'"])
def test_pin_endpoint_pin_accepts_real_kicad_pin_tokens(pin: str) -> None:
    ep = PinEndpoint(ref="U1", pin=pin)
    assert ep.pin == pin


@pytest.mark.parametrize("pin", ["pin one", "with space", "()", ""])
def test_pin_endpoint_pin_rejects_garbage(pin: str) -> None:
    with pytest.raises(ValidationError):
        PinEndpoint(ref="U1", pin=pin)


def test_pin_endpoint_ref_must_be_a_real_ref() -> None:
    with pytest.raises(ValidationError):
        PinEndpoint(ref="bad ref", pin="1")


def test_net_connection_requires_endpoints() -> None:
    with pytest.raises(ValidationError):
        NetConnection(net_name="VBUS", endpoints=[], sheet="USB INPUT")


def test_net_connection_valid_single_endpoint() -> None:
    # Hier-label-only nets can have a single in-sheet endpoint.
    nc = NetConnection(
        net_name="VBUS",
        endpoints=[PinEndpoint(ref="J1", pin="1")],
        sheet="USB INPUT",
    )
    assert nc.net_name == "VBUS"


# ---------- BOM.connections / no_connect_pins cross-validators ----------


def _two_part_bom_with_conn(**overrides) -> BOM:
    base = dict(
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
        ],
        connections=[
            NetConnection(
                net_name="VIN",
                endpoints=[
                    PinEndpoint(ref="U1", pin="1"),
                    PinEndpoint(ref="C1", pin="1"),
                ],
                sheet="REG",
            )
        ],
    )
    base.update(overrides)
    return BOM(**base)


def test_bom_connections_valid() -> None:
    bom = _two_part_bom_with_conn()
    assert len(bom.connections) == 1


def test_bom_connection_endpoint_unknown_ref_rejected() -> None:
    with pytest.raises(ValidationError):
        _two_part_bom_with_conn(
            connections=[
                NetConnection(
                    net_name="VIN",
                    endpoints=[PinEndpoint(ref="U99", pin="1")],
                    sheet="REG",
                )
            ]
        )


def test_bom_connection_unknown_sheet_rejected() -> None:
    with pytest.raises(ValidationError):
        _two_part_bom_with_conn(
            connections=[
                NetConnection(
                    net_name="VIN",
                    endpoints=[PinEndpoint(ref="U1", pin="1")],
                    sheet="GHOST",
                )
            ]
        )


def test_bom_no_connect_pin_unknown_ref_rejected() -> None:
    with pytest.raises(ValidationError):
        _two_part_bom_with_conn(no_connect_pins=[PinEndpoint(ref="U99", pin="4")])


def test_bom_no_connect_pin_valid() -> None:
    bom = _two_part_bom_with_conn(no_connect_pins=[PinEndpoint(ref="U1", pin="4")])
    assert len(bom.no_connect_pins) == 1


# ---------- ArraySpec / BOM.arrays ----------


def _led_parts(n: int) -> list[BomPart]:
    return [
        BomPart(ref=f"D{i}", value="LED", symbol="L:LED", footprint="L:LED",
                sheet="LED MATRIX")
        for i in range(1, n + 1)
    ]


def test_arrayspec_valid() -> None:
    spec = ArraySpec(refs=[f"D{i}" for i in range(1, 7)], rows=2, cols=3)
    assert spec.serpentine is True and spec.pitch_mm is None


def test_arrayspec_shape_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        ArraySpec(refs=["D1", "D2"], rows=2, cols=3)


def test_arrayspec_duplicate_refs_rejected() -> None:
    with pytest.raises(ValidationError):
        ArraySpec(refs=["D1", "D1"], rows=1, cols=2)


def test_arrayspec_nonpositive_dims_rejected() -> None:
    with pytest.raises(ValidationError):
        ArraySpec(refs=["D1"], rows=0, cols=1)


def test_bom_arrays_valid() -> None:
    bom = BOM(parts=_led_parts(4),
              arrays=[ArraySpec(refs=["D1", "D2", "D3", "D4"], rows=2, cols=2)])
    assert len(bom.arrays) == 1


def test_bom_arrays_unknown_ref_rejected() -> None:
    with pytest.raises(ValidationError):
        BOM(parts=_led_parts(2),
            arrays=[ArraySpec(refs=["D1", "D9"], rows=1, cols=2)])


def test_bom_ref_in_two_arrays_rejected() -> None:
    with pytest.raises(ValidationError):
        BOM(parts=_led_parts(3),
            arrays=[
                ArraySpec(refs=["D1", "D2"], rows=1, cols=2),
                ArraySpec(refs=["D2", "D3"], rows=1, cols=2),
            ])


def test_bom_arrays_default_empty() -> None:
    assert BOM(parts=_led_parts(1)).arrays == []
