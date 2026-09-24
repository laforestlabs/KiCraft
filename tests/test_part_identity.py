"""Physical identity boundaries for reviewed device and family requests."""

import pytest

from kicraft.design.part_identity import (
    accepted_part_identities,
    is_part_family,
    matches_part_identity,
    realizable_physical_features,
    reviewed_part,
    reviewed_parts_for_feature,
    physical_inventory_record,
)
from kicraft.design.models import BomPart

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
        ("nRF52840", "NRF52840-QIAA-R"),
        ("ULN2003", "ULN2003ADR"),
        ("ULN2003", "ULN2003A"),
        ("ULN2003A", "ULN2003ADR"),
        ("MCP23017", "MCP23017-E/SO"),
        ("STM32L0", "STM32L031K6T6"),
        ("STM32L0", "STM32L072CZU6"),
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
        ("nRF52840", "NRF52840-QIAA-R8"),
        ("ULN2003", "ULN2004ADR"),
        ("MCP23017", "MCP23S17-E/SO"),
        ("STM32L0", "STM32F103C8T6"),
        ("MAX485ESA+", "MAX485"),
        ("NRF52840-QIAA-R7", "nRF52840"),
        ("MCP23017-E/SO", "MCP23017-E/SP"),
        ("STM32L031K6T6", "STM32L031K6U6"),
        ("STM32L0", "STM32L072CBT6"),
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


def test_lowerer_provenance_requires_the_compiler_artifact_and_its_requirement():
    """A canonical lowerer part is its requirement's owner, not any group's claim."""
    requirement = {
        "id": "fpc",
        "sheet": "BREAKOUT",
        "role": "connector",
        "family": "fpc-connector",
        "parameters": {"pitch_mm": 0.5},
        "ports": {f"pin{index}": f"NET{index}" for index in range(1, 25)},
    }
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.models import CircuitRequirement

    artifact = lower_requirement(CircuitRequirement.model_validate(requirement))
    assert artifact is not None
    group = artifact.groups[0]
    part = BomPart(
        ref="J1",
        value=group.value,
        symbol=group.symbol,
        footprint=group.footprint,
        sheet="BREAKOUT",
        mpn=group.mpn,
        resolution_source="lowerer",
        resolution_id=artifact.lowerer_id,
        lowering_requirement_id="fpc",
        lowering_role=group.role,
        lowering_index=0,
    )

    assert _requirement_owns_protected_group(part, [requirement])
    assert not _requirement_owns_protected_group(
        part.model_copy(update={"lowering_requirement_id": "other"}), [requirement]
    )
    assert not _requirement_owns_protected_group(
        part.model_copy(update={"symbol": "Connector_Generic:Conn_01x24"}), [requirement]
    )


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


def test_reviewed_physical_part_metadata_requires_complete_pair():
    fpc = reviewed_part("KH-FG0.5-H2.0-24PIN")
    assert fpc is not None and fpc.is_portable_candidate
    assert fpc.contacts == tuple(str(number) for number in range(1, 25))
    assert fpc.symbol == "fpc-24-0-5-kinghelm:KH-FG0.5-H2.0-24PIN"
    assert fpc.footprint == "fpc-24-0-5-kinghelm:FPC-SMD_KH-FG0.5-H2.0-24PIN"
    capacitor = reviewed_part("C0805C103J5GACTU")
    assert capacitor is not None and capacitor.is_portable_candidate
    assert capacitor.operating_limits["dielectric"] == "C0G/NP0"
    led_driver = reviewed_part("AL8860MP-13")
    assert led_driver.current_feedback["sense_pin"] == "SET"
    assert led_driver.current_feedback["reference_pin"] == "VIN"
    assert led_driver.operating_limits["vin_min_v"] == 4.5
    posts = reviewed_parts_for_feature("binding-post")
    assert tuple(part.identity for part in posts) == ("keystone-8734",)
    jack = reviewed_part("SJ1-3533NG")
    assert jack is not None and jack.is_portable_candidate
    assert jack.port_pins == {"common": "S", "left": "T", "right": "R"}
    qwiic = reviewed_part("SM04B-SRSS-TB(LF)(SN)")
    assert qwiic is not None
    assert qwiic.contacts == ("1", "2", "3", "4")
    assert qwiic.port_pins == {"ground": "1", "vcc": "2", "sda": "3", "scl": "4"}
    assert matches_part_identity("ATTINY402", "ATTINY402-SSNR")
    assert not reviewed_part("STM32L072CZU6").is_portable_candidate
    # The corpus's "12 V DC barrel jack" shape: one reviewed, catalogued jack.
    # Its bundle is the vendored pair the loader resolves for MPN DC005, and the
    # catalog row (LCSC C431533) is the record's citation.
    jack = reviewed_part("DC005")
    assert jack is not None and jack.is_portable_candidate
    assert jack.symbol == "dc005-barrel-jack:DC005_C431533"
    assert jack.footprint == "dc005-barrel-jack:DC-IN-TH_DC005"
    assert jack.lcsc == "C431533"
    assert jack.contacts == ("1", "2", "3")
    assert "barrel-jack-connector" in jack.physical_features
    assert tuple(part.identity for part in reviewed_parts_for_feature("barrel-jack-connector")) == (
        "dc005",
    )
    # Both spellings the corpus uses resolve to that one physical class.
    assert realizable_physical_features("barrel-jack") == frozenset({"barrel-jack-connector"})
    assert realizable_physical_features("barrel-jack-connector") == frozenset(
        {"barrel-jack-connector"}
    )


def test_every_demanded_class_alias_names_an_emittable_reviewed_feature():
    """An alias must not turn a class's real-part fallback into a permanent refusal.

    `_group_has_physical_feature` refuses a demand whose class HAS reviewed coverage
    unless the BOM group resolves to a reviewed record, so an alias pointing at a
    feature no emittable record carries would block that class everywhere -- worse than
    the zero-coverage gap the alias closes. The 2026-09-24 coverage audit added aliases
    for spellings the corpus demands constantly; every target must answer with a record
    that names a symbol and a footprint.
    """
    from kicraft.design import part_identity as pi

    emittable = {
        feature
        for record in (*pi.REVIEWED_PARTS, *pi._STANDARD_LIBRARY_PARTS, *pi._STOCK_COMMON_PARTS)
        if record.symbol and record.footprint
        for feature in record.physical_features
    }
    # The stock-pattern records (`_stock_library_physical_record`) are synthesized on
    # demand from a canonical KiCad pair rather than listed in a tuple, so the features
    # they carry are emittable too.
    emittable |= (
        pi._TERMINAL_PATTERN_FEATURES
        | pi._HEADER_PATTERN_FEATURES
        | pi._LED0805_PATTERN_FEATURES
        | {pi._STACKING_HEADER_FEATURE}
    )
    for demanded, targets in pi._DEMANDED_CLASS_ALIASES.items():
        # The demand must have coverage at all ...
        assert pi.realizable_physical_features(demanded), demanded
        # ... and at least one of its targets must be a feature some record can emit.
        # (A target may legitimately be a spelling no record carries -- `header` maps to
        # `pin-socket`, which a lowerer accepts but no reviewed record declares -- as long
        # as another target does the realizing.)
        assert targets & emittable, demanded


def test_physical_inventory_is_exact_and_never_text_classified():
    header = physical_inventory_record(
        mpn=None,
        symbol="Connector_Generic:Conn_01x03",
        footprint="Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
        sourcing_note="USB connector LED audio jack",
    )
    assert header is not None
    assert header.contacts == ("1", "2", "3")
    assert (
        physical_inventory_record(
            mpn=None,
            symbol="Connector_Generic:Conn_01x03",
            footprint="Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical",
        )
        is None
    )
    assert (
        physical_inventory_record(
            mpn="NOT-A-HEADER",
            symbol="Connector_Generic:Conn_01x03",
            footprint="Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
        )
        is None
    )
    assert (
        physical_inventory_record(
            mpn="ATTINY402-SSNR",
            symbol="attiny402-ssnr:ATTINY402-SSNR",
            footprint="attiny402-ssnr:SOIC-8_L5.0-W4.0-P1.27-LS6.0-BL",
        ).identity
        == "attiny402-ssnr"
    )


def test_binding_post_catalog_order_code_requires_the_reviewed_pair():
    selected = physical_inventory_record(
        mpn="8734",
        symbol="keystone-8734-binding-post:Keystone_8734",
        footprint="keystone-8734-binding-post:Keystone_8734",
    )

    assert selected is not None and selected.identity == "keystone-8734"
    assert (
        physical_inventory_record(
            mpn="8734",
            symbol="Connector_Generic:Conn_01x01",
            footprint="Connector_PinSocket_2.54mm:PinSocket_1x01_P2.54mm_Vertical",
        )
        is None
    )
    assert (
        physical_inventory_record(
            mpn="not-keystone-8734",
            symbol="keystone-8734-binding-post:Keystone_8734",
            footprint="keystone-8734-binding-post:Keystone_8734",
        )
        is None
    )


def test_accepted_identities_are_explicit_and_package_bounded():
    assert accepted_part_identities("nRF52840") == (
        "nrf52840-qiaa-r",
        "nrf52840-qiaa-r7",
    )
    assert accepted_part_identities("STM32L0") == (
        "stm32l031k6t6",
        "stm32l072czu6",
    )
    assert accepted_part_identities("STM32L0-unreviewed") == ()


@pytest.mark.parametrize(
    ("symbol", "footprint", "contacts"),
    [
        ("Device:R", "Resistor_SMD:R_0603_1608Metric", ("1", "2")),
        ("Device:R", "Resistor_SMD:R_1210_3225Metric", ("1", "2")),
        ("Device:C", "Capacitor_SMD:C_0603_1608Metric", ("1", "2")),
        ("Device:LED", "LED_SMD:LED_0603_1608Metric", ("1", "2")),
        ("Switch:SW_Push", "Button_Switch_SMD:SW_SPST_TL3342", ("1", "2")),
        ("Connector:TestPoint", "TestPoint:TestPoint_Pad_D1.0mm", ("1",)),
        (
            "Connector:Screw_Terminal_01x03",
            "TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-3_1x03_P5.00mm_Horizontal",
            ("1", "2", "3"),
        ),
    ],
)
def test_stock_lowerer_pairs_are_bounded_exact_inventory(symbol, footprint, contacts):
    record = physical_inventory_record(mpn=None, symbol=symbol, footprint=footprint)
    assert record is not None
    assert record.contacts == contacts
    assert (
        physical_inventory_record(mpn="contradictory-mpn", symbol=symbol, footprint=footprint)
        is None
    )


def test_source_qualified_terminals_and_selector_require_exact_mpn_and_asset_pair():
    two_pin = physical_inventory_record(
        mpn="WJ126V-5.0-02P-14-00A",
        symbol="screw-terminal-5mm-2p:WJ126V-5.0-2P",
        footprint="screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
    )
    assert two_pin is not None and two_pin.contacts == ("1", "2")
    assert (
        physical_inventory_record(
            mpn="WJ126V-5.0-2P",
            symbol="screw-terminal-5mm-2p:WJ126V-5.0-2P",
            footprint="screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
        )
        is None
    )
    selector = physical_inventory_record(
        mpn="SS13D07VG4",
        symbol="ss13d07vg4:SS13D07VG4",
        footprint="ss13d07vg4:SW-TH_SS13D07VG4",
    )
    assert selector is not None
    assert selector.contacts == ("1", "2", "3", "4")
    assert selector.port_pins["common"] == "2"
    assert (
        physical_inventory_record(
            mpn="MSK13C02-SZ",
            symbol="ss13d07vg4:SS13D07VG4",
            footprint="ss13d07vg4:SW-TH_SS13D07VG4",
        )
        is None
    )


def test_reviewed_isolation_assets_have_exact_package_and_pin_contracts():
    opto = reviewed_part("PC817C-S")
    isolator = reviewed_part("ADUM1301ARWZ-RL")
    converter = reviewed_part("B0509S-1WR3")
    assert opto is not None and opto.port_pins == {
        "anode": "1",
        "cathode": "2",
        "emitter": "3",
        "collector": "4",
    }
    assert isolator is not None and isolator.port_pins["vdd1"] == "1"
    assert isolator.port_pins["vdd2"] == "16"
    assert converter is not None and converter.port_pins == {
        "input_negative": "1",
        "input_positive": "2",
        "output_negative": "3",
        "output_positive": "4",
    }


def test_source_qualified_warm_led_and_nrf_reel_identity_are_exact_assets():
    warm_led = physical_inventory_record(
        mpn="E6C0805WWAY1UDA(1.1T M)",
        symbol="e6c0805wway1uda-1-1t-m:E6C0805WWAY1UDA",
        footprint="e6c0805wway1uda-1-1t-m:LED0805-RD_WHITE",
    )
    assert warm_led is not None
    assert warm_led.port_pins == {"cathode": "1", "anode": "2"}
    assert warm_led.operating_limits["color_temperature_min_k"] == 2800
    nrf = physical_inventory_record(
        mpn="NRF52840-QIAA-R7",
        symbol="nrf52840-qiaa-r:NRF52840-QIAA-R",
        footprint="nrf52840-qiaa-r:AQFN-73_L7.0-W7.0-P0.50-BL-EP4.8",
    )
    assert nrf is not None
    assert nrf.port_pins["antenna"] == "H23"
    assert nrf.port_pins["swdio"] == "AC24"
    assert "0" in nrf.contacts
    assert (
        physical_inventory_record(
            mpn="NRF52840-QIAA-R7",
            symbol="e6c0805wway1uda-1-1t-m:E6C0805WWAY1UDA",
            footprint="e6c0805wway1uda-1-1t-m:LED0805-RD_WHITE",
        )
        is None
    )


def test_highside_pfet_requires_the_reviewed_thermal_package_and_all_power_pins():
    pfet = physical_inventory_record(
        mpn="AONR21357",
        symbol="aonr21357:AONR21357",
        footprint="aonr21357:DFN-8_L3.0-W3.0-P0.65-BL",
    )
    assert pfet is not None
    assert pfet.operating_limits["continuous_drain_a"] == 34.0
    assert pfet.support_network["source_pins"] == ("1", "2", "3")
    assert pfet.support_network["drain_pins"] == ("5", "6", "7", "8", "9")
    assert pfet.support_network["thermal_pad"] == "9"
    assert (
        physical_inventory_record(
            mpn="AONR21357",
            symbol="aonr21357:AONR21357",
            footprint="Package_SO:SO-8_3.9x4.9mm_P1.27mm",
        )
        is None
    )


def test_the_reviewed_dc_jack_maps_its_contact_functions_to_real_contacts():
    """The DC inlet's pin map comes from its datasheet schematic, not from a convention.

    Live run KC-HPD3YF could not wire the reviewed jack at all: the record held its three
    contacts but no function for any of them, so the architecture stage -- which requires a
    declared interface for a part with no curated recipe -- had nothing to derive pins from.
    Sheet 1 of the record's own datasheet draws contact 1 as the sprung tip, 2 as the
    normally-closed switch closed to 1, and 3 as the sleeve.
    """
    jack = reviewed_part("dc005")
    assert jack is not None
    assert jack.port_pins == {"tip": "1", "switch": "2", "sleeve": "3"}
    assert set(jack.port_pins.values()) <= set(jack.contacts)
    assert jack.symbol == "dc005-barrel-jack:DC005_C431533"
    assert jack.footprint == "dc005-barrel-jack:DC-IN-TH_DC005"


def test_a_contact_numbered_pin_map_names_contacts_the_record_declares():
    """A *numeric* pin map is a claim about the part's own pads, and it may not name a pad it lacks.

    The library uses two conventions, both handled by the readers: an IC maps its ports to the
    symbol's pin names (`VIN`, `SW`), a connector maps them to contact numbers (`tip` on 1). Only
    the numeric form is checkable against `contacts` without loading the symbol, and it is the one
    a connector build places directly, so a wrong number there is an unplaceable pin.
    """
    import kicraft.design.part_identity as pi

    checked = 0
    for part in pi.REVIEWED_PARTS:
        numeric = {key: value for key, value in part.port_pins.items() if str(value).isdigit()}
        if not numeric:
            continue
        checked += 1
        assert set(numeric.values()) <= set(part.contacts), part.identity
    assert checked, "no reviewed record carries a contact-numbered pin map"
