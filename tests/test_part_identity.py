"""Physical identity boundaries for reviewed device and family requests."""

import pytest

from kicraft.design.part_identity import is_part_family, matches_part_identity
from kicraft.server.stage_contracts import (
    BomComponentGroup,
    _normalize_stage_response,
    _requirement_owns_protected_group,
)


@pytest.mark.parametrize(
    ("requested", "candidate"),
    [
        ("MAX485", "MAX485ESA+"),
        ("MAX485", "MAX485ESA+T"),
        ("MAX485ESA+", "MAX485ESA+T"),
        ("nRF52840", "NRF52840-QIAA-R7"),
        ("ULN2003", "ULN2003ADR"),
        ("ULN2003", "ULN2003A"),
        ("ULN2003A", "ULN2003ADR"),
        ("MCP23017", "MCP23017-E/SO"),
        ("STM32L0", "STM32L031K6T6"),
    ],
)
def test_reviewed_candidates_implement_requested_identity(requested, candidate):
    group = _group(candidate, value=requested)
    assert matches_part_identity(requested, candidate)
    assert _requirement_owns_protected_group(group, [{"exact_part": requested}])


@pytest.mark.parametrize(
    ("requested", "candidate"),
    [
        ("MAX485", "MAX3485"),
        ("nRF52840", "NRF52832-QFAA-R7"),
        ("nRF52840", "MDBT50Q-1MV2"),
        ("ULN2003", "ULN2004ADR"),
        ("MCP23017", "MCP23S17-E/SO"),
        ("STM32L0", "STM32F103C8T6"),
        ("MAX485ESA+", "MAX485"),
        ("NRF52840-QIAA-R7", "nRF52840"),
        ("MCP23017-E/SO", "MCP23017-E/SP"),
        ("STM32L031K6T6", "STM32L031K6U6"),
        ("MAX485", "MAX485ESA+UNREVIEWED"),
        ("MAX485ESA+", "MAX485ESA"),
        ("BME280", "K2"),
        ("STM32L0", "STM32L0"),
        ("STM32L0", "STM32L0-controller"),
    ],
)
def test_wrong_or_unreviewed_identity_cannot_hide_behind_matching_value(requested, candidate):
    assert not matches_part_identity(requested, candidate)
    assert not _requirement_owns_protected_group(
        _group(candidate, value=requested),
        [{"id": "controller", "family": requested, "exact_part": requested}],
    )


def _group(mpn, *, value="STM32L0"):
    return BomComponentGroup(
        id="controller",
        reference_prefix="U",
        quantity=1,
        value=value,
        mpn=mpn,
        symbol="Device:U",
        footprint="Package_QFP:LQFP-32",
        sheet="MCU",
    )


def test_family_requirement_needs_member_even_when_group_label_matches():
    requirement = {"id": "controller", "family": "stm32l0"}
    assert is_part_family(" STM32L0 ")
    assert not is_part_family("STM32L031K6T6")
    assert _requirement_owns_protected_group(_group("STM32L031K6T6"), [requirement])
    assert not _requirement_owns_protected_group(_group("STM32F103C8T6"), [requirement])
    assert not _requirement_owns_protected_group(_group("STM32L0"), [requirement])
    assert not _requirement_owns_protected_group(_group(None), [requirement])


def test_equality_keeps_case_insensitivity_but_not_order_code_punctuation_erasure():
    assert matches_part_identity(" Unknown-MPN/1 ", "unknown-mpn/1")
    assert not matches_part_identity("Unknown-MPN/1", "UnknownMPN1")
    assert not matches_part_identity("", "")
    assert not matches_part_identity("  ", " ")


def test_architecture_family_is_not_stamped_as_physical_part():
    architecture, _ = _normalize_stage_response(
        "architecture",
        {
            "sheets": [{"name": "MCU", "stem": "MCU", "function": "controller"}],
            "power_nets": [],
            "inter_sheet_nets": [],
            "requirements": [
                {"id": "controller", "sheet": "MCU", "role": "mcu_core", "family": "stm32l0"}
            ],
        },
        {"intent": {"named_parts": ["STM32L0"]}},
    )
    requirement = architecture["requirements"][0]
    assert requirement.get("exact_part") is None
    assert _requirement_owns_protected_group(_group("STM32L031K6T6"), [requirement])
    assert not _requirement_owns_protected_group(_group("STM32L0"), [requirement])
