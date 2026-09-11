import json
import math

import pytest

from kicraft.design.lowering import lower_requirement
from kicraft.design.models import CircuitRequirement
from kicraft.server.stage_work_units import (
    StageWorkUnit,
    deterministic_bom_candidate,
    deterministic_wiring_candidate,
    merge_bom_units,
    plan_stage_work_units,
    validate_unit_candidate,
)


def _requirement(family: str, *, parameters=None, ports=None) -> CircuitRequirement:
    return CircuitRequirement(
        id="block",
        sheet="MAIN",
        role="analog_block",
        family=family,
        parameters=parameters or {},
        ports=ports or {},
    )


def _state(requirement: CircuitRequirement, parts=None) -> dict:
    return {
        "architecture": {
            "sheets": [{"name": "MAIN"}],
            "requirements": [requirement.model_dump(mode="json")],
            "recipe_selections": [],
            "power_nets": ["GND"],
            "inter_sheet_nets": [],
        },
        "bom": {"parts": parts or []},
    }


def test_verified_lowerer_can_own_an_exact_part_used_inside_an_active_recipe():
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution

    requirement = CircuitRequirement(
        id="usb",
        sheet="USB",
        role="connector",
        family="usb-c-breakout",
        exact_part="TYPE-C-31-M-12",
        ports={"vbus": "VBUS", "gnd": "GND"},
    )
    architecture = {
        "topologies": {},
        "rail_voltages": {"VBUS": 5.0},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [{"name": "USB", "stem": "USB", "function": "Passive receptacle"}],
        "power_nets": ["VBUS", "GND"],
        "inter_sheet_nets": [],
        "assumptions": [],
        "requirements": [requirement.model_dump()],
    }
    resolved = apply_architecture_recipe_resolution(architecture)
    assert resolved.recipe_selections == []
    artifact = lower_requirement(resolved.requirements[0])
    assert [group.reference_prefix for group in artifact.groups] == ["J"]
    assert {"A5", "B5"} <= {pin.pin for pin in artifact.no_connects}


def _pinout(count: int) -> dict:
    return {"pins": [{"number": str(index)} for index in range(1, count + 1)]}


def test_passive_usb_breakout_carries_exact_symbol_pins_without_terminations():
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    requirement = CircuitRequirement(
        id="usb",
        sheet="MAIN",
        role="connector",
        family="usb-c-breakout",
        ports={
            "vbus": "VBUS",
            "gnd": "GND",
            "usb_dp": "D+",
            "usb_dm": "D-",
            "cc1": "CC1",
            "cc2": "CC2",
            "sbu1": "SBU1",
            "sbu2": "SBU2",
        },
    )
    artifact = lower_requirement(requirement)
    assert [group.reference_prefix for group in artifact.groups] == ["J"]
    by_pin = {pin.pin: pin.net for pin in artifact.pins}
    assert len(artifact.pins) == len(by_pin)
    assert set(by_pin) | {pin.pin for pin in artifact.no_connects} == {
        pin["number"] for pin in lookup_pins(artifact.groups[0].symbol, all_units=True)["pins"]
    }
    assert {by_pin["A6"], by_pin["B6"]} == {"D+"}
    assert {by_pin["A7"], by_pin["B7"]} == {"D-"}
    assert {by_pin["A1B12"], by_pin["B1A12"]} == {"GND"}
    assert {by_pin["A4B9"], by_pin["B4A9"]} == {"VBUS"}
    assert by_pin["A5"] == "CC1"
    assert by_pin["B5"] == "CC2"
    assert {pin.pin for pin in artifact.no_connects} == {"1", "2", "3", "4"}
    assert not any(pin in by_pin for pin in ("1", "2", "3", "4"))
    wider = requirement.model_copy(update={"ports": {**requirement.ports, "tx1p": "TX1+"}})
    state = _state(wider)
    state["architecture"]["sheets"][0]["function"] = "USB-C receptacle breakout"
    unit = StageWorkUnit("bom-r000", "bom", "MAIN", requirement_ids=("usb",))
    candidate = deterministic_bom_candidate(unit, state)
    assert candidate["groups"][0]["symbol"] == "Connector:USB_C_Receptacle"


def test_superspeed_breakout_uses_real_contacts_without_usb2_substitution():
    from pathlib import Path

    import pcbnew

    from kicraft.design.synthesis.footprint_library import load_footprint
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    ports = {
        "vbus": "VBUS",
        "gnd": "GND",
        "cc1": "CC1",
        "cc2": "CC2",
        "sbu1": "SBU1",
        "sbu2": "SBU2",
        "usb_dp": "D+",
        "usb_dm": "D-",
        "tx1p": "TX1+",
        "tx1n": "TX1-",
        "tx2p": "TX2+",
        "tx2n": "TX2-",
        "rx1p": "RX1+",
        "rx1n": "RX1-",
        "rx2p": "RX2+",
        "rx2n": "RX2-",
    }
    artifact = lower_requirement(_requirement("usb-c-breakout", ports=ports))
    group = artifact.groups[0]
    assert group.mpn == "12401610E4#2A"
    connected = {pin.pin: pin.net for pin in artifact.pins}
    expected = {
        "A1": "GND",
        "A2": "TX1+",
        "A3": "TX1-",
        "A4": "VBUS",
        "A5": "CC1",
        "A6": "D+",
        "A7": "D-",
        "A8": "SBU1",
        "A9": "VBUS",
        "A10": "RX2-",
        "A11": "RX2+",
        "A12": "GND",
        "B1": "GND",
        "B2": "TX2+",
        "B3": "TX2-",
        "B4": "VBUS",
        "B5": "CC2",
        "B6": "D+",
        "B7": "D-",
        "B8": "SBU2",
        "B9": "VBUS",
        "B10": "RX1-",
        "B11": "RX1+",
        "B12": "GND",
    }
    assert connected == expected
    assert {pin.pin for pin in artifact.no_connects} == {"S1"}
    symbol = lookup_pins(group.symbol, all_units=True)
    footprint, _ = load_footprint(pcbnew, *group.footprint.split(":"), project_root=Path("."))
    owned = set(connected) | {pin.pin for pin in artifact.no_connects}
    assert {pad.GetNumber() for pad in footprint.Pads() if pad.GetNumber()} == owned
    assert {pin["number"] for pin in symbol["pins"]} == owned
    assert {
        pin["number"]: pin["name"]
        for pin in symbol["pins"]
        if pin["number"] in ("A2", "A3", "B2", "B3", "A10", "A11", "B10", "B11")
    } == {pin: expected[pin] for pin in ("A2", "A3", "B2", "B3", "A10", "A11", "B10", "B11")}


@pytest.mark.parametrize("exposed", [{}, {"tx1p": "TX1+", "tx1n": "TX1-"}])
def test_exact_superspeed_subset_disconnects_every_omitted_contact(exposed):
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    requirement = _requirement(
        "usb-c-breakout", ports={"vbus": "VBUS", "gnd": "GND", **exposed}
    ).model_copy(update={"exact_part": "12401610E4#2A"})
    artifact = lower_requirement(requirement)
    connected = {pin.pin: pin.net for pin in artifact.pins}
    disconnected = {pin.pin for pin in artifact.no_connects}
    expected_nc = {
        "S1",
        "A2",
        "A3",
        "B2",
        "B3",
        "A10",
        "A11",
        "B10",
        "B11",
        "A5",
        "B5",
        "A6",
        "B6",
        "A7",
        "B7",
        "A8",
        "B8",
    } - ({"A2", "A3"} if exposed else set())
    assert disconnected == expected_nc
    assert set(connected).isdisjoint(disconnected)
    assert set(connected) | disconnected == {
        pin["number"] for pin in lookup_pins(artifact.groups[0].symbol, all_units=True)["pins"]
    }
    if exposed:
        assert connected["A2"] == "TX1+"
        assert connected["A3"] == "TX1-"
    state = _state(
        requirement,
        parts=[
            {
                "ref": "J1",
                "sheet": "MAIN",
                "symbol": artifact.groups[0].symbol,
                "resolution_source": "lowerer",
                "resolution_id": artifact.lowerer_id,
                "lowering_requirement_id": requirement.id,
                "lowering_role": "connector",
            }
        ],
    )
    extras = {"symbol_pinouts": {"J1": lookup_pins(artifact.groups[0].symbol, all_units=True)}}
    (unit,) = plan_stage_work_units("wiring", state, extras)
    candidate = deterministic_wiring_candidate(unit, state, extras)
    assert {row["pin"] for row in candidate["pins"] if row.get("no_connect")} == disconnected
    assert {row["pin"]: row["net"] for row in candidate["pins"] if row.get("net")} == connected


@pytest.mark.parametrize("exact_part", ["TYPE-C-31-M-12", "USB4085", "12401610E4-2A"])
def test_superspeed_never_substitutes_for_another_exact_connector(exact_part):
    requirement = _requirement(
        "usb-c-breakout", ports={"vbus": "VBUS", "gnd": "GND", "tx1p": "TX1+"}
    ).model_copy(update={"exact_part": exact_part})
    assert lower_requirement(requirement) is None


@pytest.mark.parametrize(
    "ports",
    [
        {"vbus": "VBUS", "gnd": "GND"},
        {"vbus": "VBUS", "gnd": "GND", "cc1": "PD_CC1", "cc2": "PD_CC2"},
    ],
)
def test_passive_usb_subset_explicitly_disconnects_unexposed_signal_pins(ports):
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    requirement = _requirement("usb-c-breakout", ports=ports)
    artifact = lower_requirement(requirement)
    assert [group.reference_prefix for group in artifact.groups] == ["J"]
    connected = {pin.pin: pin.net for pin in artifact.pins}
    disconnected = {pin.pin for pin in artifact.no_connects}
    assert disconnected == {"1", "2", "3", "4", "A6", "B6", "A7", "B7", "A8", "B8"} | (
        set() if "cc1" in ports else {"A5", "B5"}
    )
    assert set(connected).isdisjoint(disconnected)
    assert set(connected) | disconnected == {
        pin["number"] for pin in lookup_pins(artifact.groups[0].symbol, all_units=True)["pins"]
    }
    if "cc1" in ports:
        assert connected["A5"] == "PD_CC1"
        assert connected["B5"] == "PD_CC2"

    state = _state(requirement)
    state["bom"]["parts"] = [
        {
            "ref": "J1",
            "sheet": "MAIN",
            "symbol": artifact.groups[0].symbol,
            "resolution_source": "lowerer",
            "resolution_id": artifact.lowerer_id,
            "lowering_requirement_id": requirement.id,
            "lowering_role": "connector",
        }
    ]
    extras = {"symbol_pinouts": {"J1": lookup_pins(artifact.groups[0].symbol, all_units=True)}}
    (unit,) = plan_stage_work_units("wiring", state, extras)
    candidate = deterministic_wiring_candidate(unit, state, extras)
    assert {row["pin"] for row in candidate["pins"] if row.get("no_connect")} == disconnected
    assert {row["pin"]: row["net"] for row in candidate["pins"] if row.get("net")} == connected


@pytest.mark.parametrize(
    ("exact_part", "shell_pins"),
    [("TYPE-C-31-M-12", {"1", "2", "3", "4"}), ("12401610E4#2A", {"S1"})],
)
def test_explicit_usb_shield_binding_is_not_grounded_or_disconnected(exact_part, shell_pins):
    requirement = _requirement(
        "usb-c-breakout", ports={"vbus": "VBUS", "gnd": "GND", "shield": "CHASSIS"}
    ).model_copy(update={"exact_part": exact_part})
    artifact = lower_requirement(requirement)
    connected = {pin.pin: pin.net for pin in artifact.pins}
    assert {connected[pin] for pin in shell_pins} == {"CHASSIS"}
    assert shell_pins.isdisjoint(pin.pin for pin in artifact.no_connects)

    grounded = lower_requirement(
        requirement.model_copy(update={"ports": {**requirement.ports, "shield": "GND"}})
    )
    assert {pin.net for pin in grounded.pins if pin.pin in shell_pins} == {"GND"}


@pytest.mark.parametrize(
    "ports",
    [
        {"vbus": "VBUS"},
        {"vbus": "VBUS", "gnd": "GND", "unknown_lane": "TX+"},
        {"vbus": "VBUS", "gnd": "GND", "cc1": ""},
    ],
)
def test_passive_usb_lowerer_refuses_unproven_or_unsupported_bindings(ports):
    assert lower_requirement(_requirement("usb-c-breakout", ports=ports)) is None


def test_lowerer_registry_matches_exact_family_without_ordered_fallback():
    assert lower_requirement(_requirement("pin_header_extra", ports={"one": "N"})) is None
    assert lower_requirement(_requirement("unknown", ports={"one": "N"})) is None
    assert lower_requirement(_requirement("pin-header", ports={"pin1": "N"})) is not None


@pytest.mark.parametrize(("rows", "count"), [(1, 4), (2, 2), (2, 8)])
def test_pin_header_lowerer_preserves_real_contact_geometry(rows, count):
    from pathlib import Path

    import pcbnew

    from kicraft.design.synthesis.footprint_library import load_footprint
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    requirement = _requirement(
        "pin-header",
        parameters={"rows": rows},
        ports={f"pin{index}": f"NET{index}" for index in range(1, count + 1)},
    )
    artifact = lower_requirement(requirement)
    group = artifact.groups[0]
    symbol = lookup_pins(group.symbol, all_units=True)
    footprint, _ = load_footprint(pcbnew, *group.footprint.split(":"), project_root=Path("."))
    contacts = {str(index) for index in range(1, count + 1)}
    assert {pin["number"] for pin in symbol["pins"]} == contacts
    assert {pad.GetNumber() for pad in footprint.Pads()} == contacts
    assert {pin.pin for pin in artifact.pins} == contacts
    assert len({pad.GetPosition().x for pad in footprint.Pads()}) == rows


def test_reserved_header_contacts_are_no_connects_not_a_shared_net():
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    requirement = _requirement(
        "pin-header",
        ports={"pin1": "NC", "pin2": "SIGNAL", "pin3": "GND", "pin4": "NC"},
    )
    artifact = lower_requirement(requirement)
    state = _state(
        requirement,
        parts=[
            {
                "ref": "J1",
                "sheet": "MAIN",
                "symbol": artifact.groups[0].symbol,
                "resolution_source": "lowerer",
                "resolution_id": artifact.lowerer_id,
                "lowering_requirement_id": requirement.id,
                "lowering_role": "connector",
            }
        ],
    )
    extras = {"symbol_pinouts": {"J1": lookup_pins(artifact.groups[0].symbol, all_units=True)}}
    (unit,) = plan_stage_work_units("wiring", state, extras)
    candidate = deterministic_wiring_candidate(unit, state, extras)
    assert {row["pin"] for row in candidate["pins"] if row.get("no_connect")} == {"1", "4"}
    assert {row["pin"]: row["net"] for row in candidate["pins"] if row.get("net")} == {
        "2": "SIGNAL",
        "3": "GND",
    }


@pytest.mark.parametrize("rows", [16, 1.5, True, "2"])
def test_header_lowerer_does_not_coerce_unsupported_row_choices(rows):
    requirement = _requirement(
        "pin-header",
        parameters={"rows": rows},
        ports={f"pin{index}": f"NET{index}" for index in range(1, 17)},
    )
    assert lower_requirement(requirement) is None


@pytest.mark.parametrize("family", ["pin-header", "fpc-header-breakout"])
def test_header_physical_numbers_survive_sorted_json_round_trip(family):
    requirement = _requirement(
        family, ports={f"pin{index}": f"NET{index}" for index in range(1, 17)}
    )
    reordered = json.loads(json.dumps(requirement.model_dump(mode="json"), sort_keys=True))
    roles = ("connector",) if family == "pin-header" else ("fpc", "header")
    expected = {(role, str(index)): f"NET{index}" for role in roles for index in range(1, 17)}
    assert {(pin.role, pin.pin): pin.net for pin in lower_requirement(requirement).pins} == expected
    assert {(pin.role, pin.pin): pin.net for pin in lower_requirement(reordered).pins} == expected


@pytest.mark.parametrize(
    "ports",
    [
        {"one": "NET1", "two": "NET2"},
        {"pin1": "NET1", "pin3": "NET3"},
        {"pin0": "NET0", "pin1": "NET1"},
        {"pin01": "NET1", "pin2": "NET2"},
        {"pin1": "NET1", "signal": "NET2"},
        {"pin1": "NET1", "pin2": " "},
    ],
)
def test_header_rejects_ambiguous_or_gapped_contact_contracts(ports):
    assert lower_requirement(_requirement("pin-header", ports=ports)) is None


@pytest.mark.parametrize(
    "updates",
    [
        {"parameters": {"pin_count": 20}},
        {"parameters": {"pitch_mm": 1.27}},
        {"exact_part": "TSW-118-07-L-D"},
    ],
)
def test_header_physical_constraints_remain_model_owned_when_unproven(updates):
    requirement = _requirement("pin-header", ports={"pin1": "SIGNAL", "pin2": "GND"}).model_copy(
        update=updates
    )
    assert lower_requirement(requirement) is None
    unit = StageWorkUnit(stage="bom", unit_id="bom:MAIN", sheet="MAIN", requirement_ids=("block",))
    state = _state(requirement)
    state["architecture"]["sheets"][0]["function"] = "2-pin header"
    assert deterministic_bom_candidate(unit, state) is None


def test_screw_terminal_honors_registered_row_parameter():
    requirement = _requirement(
        "screw-terminal", parameters={"rows": 1}, ports={"positive": "VIN", "negative": "GND"}
    )
    artifact = lower_requirement(requirement)
    assert artifact is not None
    assert artifact.groups[0].symbol == "Connector:Screw_Terminal_01x02"
    assert lower_requirement(requirement.model_copy(update={"parameters": {"rows": 2}})) is None


def test_curated_selector_proves_only_its_existing_part_identity():
    requirement = _requirement(
        "voltage_selector_switch",
        parameters={"positions": 3},
        ports={"SEL0": "SEL0", "SEL1": "SEL1"},
    )
    artifact = lower_requirement(requirement.model_copy(update={"exact_part": "MSK13C02-SZ"}))
    assert artifact is not None
    assert artifact.groups[0].mpn == "MSK13C02-SZ"
    assert lower_requirement(requirement.model_copy(update={"exact_part": "MSK12C02"})) is None


def test_switch_lowerer_does_not_guess_an_absent_pull_policy():
    requirement = _requirement(
        "switch-input",
        parameters={"active_level": "high", "resistance": 10000},
        ports={"signal": "BOOT0", "gnd": "GND", "vdd": "+3V3"},
    )
    assert lower_requirement(requirement) is None


def test_internal_switch_bias_cannot_discard_requested_resistor():
    requirement = _requirement(
        "switch-input",
        parameters={"pull_policy": "internal", "resistance": "4.7k"},
        ports={"signal": "BUTTON", "gnd": "GND", "vdd": "3V3"},
    )
    assert lower_requirement(requirement) is None


def test_r2r_conflicting_parameter_aliases_remain_model_owned():
    requirement = _requirement(
        "r2r-ladder",
        parameters={"bits": 2, "r_value": "10k", "r_series": "22k", "two_r_value": "20k"},
        ports={"bit0": "D0", "bit1": "D1", "output": "OUT", "gnd": "GND"},
    )
    assert lower_requirement(requirement) is None


def test_voltage_selector_lowerer_uses_curated_compatible_pair():
    artifact = lower_requirement(
        _requirement(
            "voltage_selector_switch",
            parameters={"positions": 3},
            ports={"SEL0": "SEL0", "SEL1": "SEL1"},
        )
    )

    assert artifact is not None
    assert artifact.lowerer_id == "voltage-selector-switch@1"
    assert artifact.groups[0].symbol == "sp3t-switch-msk13c02:MSK13C02-SZ"
    assert artifact.groups[0].footprint == ("sp3t-switch-msk13c02:SW-SMD_MSK13C02-SZ")


@pytest.mark.parametrize("policy", ["internal", "external"])
@pytest.mark.parametrize(
    ("active_level", "active_net", "pull_role", "idle_net"),
    [("low", "GND", "pullup", "3V3"), ("high", "3V3", "pulldown", "GND")],
)
def test_button_actuation_and_bias_follow_explicit_polarity(
    policy, active_level, active_net, pull_role, idle_net
):
    artifact = lower_requirement(
        _requirement(
            "switch-input",
            parameters={"pull_policy": policy, "active_level": active_level},
            ports={"signal": "CONTROL", "vdd": "3V3", "gnd": "GND"},
        )
    )
    by_pin = {(pin.role, pin.pin): pin.net for pin in artifact.pins}
    expected = {
        ("switch", "1"): "CONTROL",
        ("switch", "2"): active_net,
    }
    if policy == "external":
        expected.update({(pull_role, "1"): idle_net, (pull_role, "2"): "CONTROL"})
    assert by_pin == expected
    assert {group.role for group in artifact.groups} == (
        {"switch", pull_role} if policy == "external" else {"switch"}
    )
    assert next(group for group in artifact.groups if group.role == "switch").symbol == (
        "Switch:SW_Push"
    )


@pytest.mark.parametrize(
    ("parameters", "ports"),
    [
        ({"pull_policy": "external"}, {"signal": "CONTROL", "gnd": "GND"}),
        (
            {"pull_policy": "external"},
            {"signal": "CONTROL", "gnd": "GND", "vdd": "GND"},
        ),
        (
            {"pull_policy": "external", "active_level": "toggle"},
            {"signal": "CONTROL", "gnd": "GND", "vdd": "3V3"},
        ),
    ],
)
def test_button_lowerer_refuses_ambiguous_supply_or_actuation(parameters, ports):
    assert (
        lower_requirement(_requirement("switch-input", parameters=parameters, ports=ports)) is None
    )


def test_coin_cell_holder_lowers_real_source_and_preserves_battery_polarity():
    from pathlib import Path

    import pcbnew

    from kicraft.design.synthesis.footprint_library import load_footprint
    from kicraft.design.synthesis.symbol_pinout import lookup_pins
    from kicraft.server.stage_contracts import _expand_bom_groups

    requirement = _requirement(
        "coin-cell-holder",
        parameters={"cell_format": "CR2032"},
        ports={"positive": "VBAT", "negative": "RETURN"},
    )
    state = _state(requirement)
    (bom_unit,) = plan_stage_work_units("bom", state, {})
    candidate = validate_unit_candidate(bom_unit, {"groups": []}, state, {})
    merged, _, provenance, _ = merge_bom_units((bom_unit,), {bom_unit.unit_id: candidate}, state)
    _, parts, _, _, _ = _expand_bom_groups(merged, state)
    (holder,) = parts
    assert holder.mpn == "BS-07-A1BJ001"
    assert holder.footprint == "Battery:BatteryHolder_MYOUNG_BS-07-A1BJ001_CR2032"
    assert holder.sourcing_note.startswith("LCSC C2979167;")
    footprint, _ = load_footprint(pcbnew, *holder.footprint.split(":"), project_root=Path("."))
    assert {pad.GetNumber() for pad in footprint.Pads()} == {"1", "2"}
    pinout = lookup_pins(holder.symbol, project_root=Path("."))
    assert {pin["number"]: pin["name"] for pin in pinout["pins"]} == {"1": "+", "2": "-"}
    state["bom"]["parts"] = [
        {
            **holder.model_dump(mode="json"),
            "resolution_source": "lowerer",
            **provenance[holder.ref],
        }
    ]
    extras = {"symbol_pinouts": {holder.ref: pinout}}
    (unit,) = plan_stage_work_units("wiring", state, extras)
    wiring = deterministic_wiring_candidate(unit, state, extras)
    assert {(row["ref"], row["pin"]): row["net"] for row in wiring["pins"]} == {
        (holder.ref, "1"): "VBAT",
        (holder.ref, "2"): "RETURN",
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"parameters": {"cell_format": "CR2025"}},
        {"parameters": {}},
        {"parameters": {"cell_format": "CR2032", "cells": 2}},
        {"ports": {"positive": "VBAT"}},
        {"ports": {"positive": "VBAT", "negative": "VBAT"}},
        {"ports": {"positive": "VBAT", "negative": "GND", "output": "VBAT"}},
        {"exact_part": "CR2032-HFN"},
        {"parameters": {"cell_format": "CR2032", "named_part_identities": "CR2032"}},
    ],
)
def test_coin_cell_holder_refuses_other_cells_and_ambiguous_bindings(updates):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    requirement = _requirement(
        "coin-cell-holder",
        parameters={"cell_format": "CR2032"},
        ports={"positive": "VBAT", "negative": "GND"},
    )
    assert lower_requirement(requirement.model_copy(update=updates)) is None

    invalid = requirement.model_copy(update=updates)
    result = resolve_architecture_recipes(
        {
            "sheets": [{"name": "MAIN", "stem": "MAIN", "function": "Battery source"}],
            "requirements": [invalid],
            "topologies": {},
            "rail_voltages": {"VBAT": 3.0},
            "power_nets": ["VBAT", "GND"],
            "comms_protocols": [],
            "mcu_present": False,
            "inter_sheet_nets": [],
            "assumptions": [],
        },
        {"named_parts": ["CR2032"]},
    )
    assert "missing_recipe_requirement" in {row.code for row in result.blocking}


@pytest.mark.parametrize(
    "family",
    [
        "resistor-network",
        "resistor_network",
        "resistor-ladder",
        "resistor_ladder",
        "r2r_ladder",
    ],
)
def test_r2r_lowerer_accepts_compact_architecture_requirement(family):
    artifact = lower_requirement(
        _requirement(
            family,
            parameters={"bits": 8, "r_series": 10000, "r_shunt": 20000},
            ports={"digital_inputs": "D0-D7", "analog_output": "LADDER_OUT"},
        )
    )

    assert artifact is not None
    assert artifact.lowerer_id == "r2r-ladder@1"
    assert [(group.quantity, group.value) for group in artifact.groups] == [
        (7, "10k"),
        (9, "20k"),
    ]
    assert len(artifact.pins) == 32
    nets = {pin.net for pin in artifact.pins}
    assert {"D0", "D7", "LADDER_OUT", "GND"} <= nets


def test_r2r_lowerer_derives_two_r_from_single_r_parameter():
    artifact = lower_requirement(
        _requirement(
            "resistor_network",
            parameters={"bits": 8, "r": 10000},
            ports={"digital_inputs": "D0-D7", "analog_output": "LADDER_OUT"},
        )
    )

    assert artifact is not None
    assert [(group.quantity, group.value) for group in artifact.groups] == [
        (7, "10k"),
        (9, "20k"),
    ]


@pytest.mark.parametrize(
    ("family", "parameters", "ports"),
    [
        (
            "led-current-resistor",
            {"rail_voltage": 3.3, "led_vf": 2.0, "target_current_ma": 5},
            {"drive": "STATUS", "gnd": "GND"},
        ),
        (
            "voltage-divider",
            {"input_voltage": 12, "target_voltage": 3.3, "bottom_resistance_ohm": 10000},
            {"input": "VIN", "output": "SENSE", "gnd": "GND"},
        ),
        (
            "rc-lowpass",
            {"cutoff_hz": 1000, "resistance_ohm": 10000},
            {"input": "RAW", "output": "FILTERED", "gnd": "GND"},
        ),
    ],
)
def test_calculated_lowerers_choose_standard_values_with_bounded_error(family, parameters, ports):
    artifact = lower_requirement(_requirement(family, parameters=parameters, ports=ports))
    assert artifact is not None
    assert artifact.calculations
    assert "unrounded" not in artifact.calculations[0].tolerance


def test_voltage_divider_rejects_unachievable_error_bound():
    requirement = _requirement(
        "voltage-divider",
        parameters={
            "input_voltage": 12,
            "target_voltage": math.pi,
            "bottom_resistance_ohm": 10000,
            "max_error_percent": 0.001,
        },
        ports={"input": "VIN", "output": "SENSE", "gnd": "GND"},
    )
    assert lower_requirement(requirement) is None


@pytest.mark.parametrize("capacitance", [10, 400])
def test_i2c_pullup_lowerer_accepts_bus_capacitance_boundaries(capacitance):
    requirement = _requirement(
        "i2c-pullups",
        parameters={"speed_hz": 400000, "bus_capacitance_pf": capacitance, "voltage": 3.3},
        ports={"vdd": "3V3", "sda": "SDA", "scl": "SCL"},
    )
    assert lower_requirement(requirement) is not None


@pytest.mark.parametrize("capacitance", [9, 401])
def test_i2c_pullup_lowerer_rejects_out_of_range_capacitance(capacitance):
    requirement = _requirement(
        "i2c-pullups",
        parameters={"speed_hz": 400000, "bus_capacitance_pf": capacitance, "voltage": 3.3},
        ports={"vdd": "3V3", "sda": "SDA", "scl": "SCL"},
    )
    assert lower_requirement(requirement) is None


def test_r2r_bom_provenance_drives_wiring_from_the_same_artifact():
    requirement = _requirement(
        "r2r-ladder",
        parameters={"bits": 3, "r_value": "10k", "two_r_value": "20k"},
        ports={"bit0": "D0", "bit1": "D1", "bit2": "D2", "output": "DAC", "gnd": "GND"},
    )
    state = _state(requirement)
    bom_unit = StageWorkUnit("bom-r000", "bom", "MAIN", requirement_ids=(requirement.id,))
    candidate = validate_unit_candidate(bom_unit, {"groups": [], "arrays": []}, state, {})
    merged, ref_to_unit, ref_to_lowering, trusted_groups = merge_bom_units(
        (bom_unit,), {bom_unit.unit_id: candidate}, state
    )
    assert trusted_groups == frozenset({"r000_branch", "r000_series"})
    assert ref_to_unit == {f"R{index}": "bom-r000" for index in range(1, 7)}
    assert ref_to_lowering["R1"] == {
        "resolution_id": "r2r-ladder@1",
        "lowering_requirement_id": "block",
        "lowering_role": "series",
        "lowering_index": 0,
    }

    parts = []
    next_ref = 1
    for group in merged["groups"]:
        for _ in range(group["quantity"]):
            ref = f"R{next_ref}"
            next_ref += 1
            parts.append(
                {
                    "ref": ref,
                    "sheet": "MAIN",
                    "symbol": group["symbol"],
                    "value": group["value"],
                    "resolution_source": "lowerer",
                    **ref_to_lowering[ref],
                }
            )
    state["bom"] = {"parts": parts}
    extras = {"symbol_pinouts": {part["ref"]: _pinout(2) for part in parts}}
    units = plan_stage_work_units("wiring", state, extras)
    assert len(units) == 1
    assert units[0].planned_resolution_source == "lowerer"
    assert units[0].lowerer_ids == ("r2r-ladder@1",)

    wiring = deterministic_wiring_candidate(units[0], state, extras)
    assert wiring is not None
    by_pin = {(row["ref"], row["pin"]): row["net"] for row in wiring["pins"]}
    assert by_pin[("R1", "1")] == "DAC"
    assert by_pin[("R3", "1")] == "D2"
    assert by_pin[("R6", "2")] == "GND"


def test_bom_candidate_requires_typed_requirement_ownership():
    requirement = _requirement("pin-header", ports={"pin1": "GND"})
    state = _state(requirement)
    assert deterministic_bom_candidate(StageWorkUnit("bom-s000", "bom", "MAIN"), state) is None


def test_lowerer_groups_use_loadable_symbols_and_footprints():
    from pathlib import Path

    from kicraft.design.synthesis.footprint_library import lookup_footprint
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    requirements = [
        _requirement("pin-header", parameters={"rows": 1}, ports={"pin1": "GND", "pin2": "3V3"}),
        _requirement("screw-terminal", ports={"p0": "VIN", "p1": "GND"}),
        _requirement(
            "fpc-header-breakout",
            parameters={"pitch_mm": 0.5},
            ports={f"pin{index}": f"N{index}" for index in range(1, 11)},
        ),
        _requirement("test-points", ports={"a": "A", "b": "B"}),
        _requirement(
            "connector-bank",
            parameters={"channels": 2},
            ports={"vdd": "5V", "gnd": "GND", "signal0": "S0", "signal1": "S1"},
        ),
        _requirement(
            "r2r-ladder",
            parameters={"bits": 2, "r_value": "10k", "two_r_value": "20k"},
            ports={"bit0": "D0", "bit1": "D1", "output": "DAC", "gnd": "GND"},
        ),
        _requirement(
            "led-current-resistor",
            parameters={"rail_voltage": 3.3, "led_vf": 2.0, "target_current_ma": 5},
            ports={"drive": "LED", "gnd": "GND"},
        ),
        _requirement(
            "voltage-divider",
            parameters={"input_voltage": 12, "target_voltage": 3.3, "bottom_resistance_ohm": 10000},
            ports={"input": "VIN", "output": "SENSE", "gnd": "GND"},
        ),
        _requirement(
            "rc-lowpass",
            parameters={"cutoff_hz": 1000, "resistance_ohm": 10000},
            ports={"input": "RAW", "output": "FILTERED", "gnd": "GND"},
        ),
        _requirement(
            "i2c-pullups",
            parameters={"speed_hz": 400000, "bus_capacitance_pf": 100, "voltage": 3.3},
            ports={"vdd": "3V3", "sda": "SDA", "scl": "SCL"},
        ),
        _requirement(
            "open-drain-pullup",
            parameters={"resistance": "10k"},
            ports={"vdd": "3V3", "signal": "IRQ"},
        ),
        _requirement(
            "switch-input",
            parameters={"pull_policy": "external", "resistance": "10k"},
            ports={"signal": "BUTTON", "gnd": "GND", "vdd": "3V3"},
        ),
        _requirement(
            "explicit-decoupling",
            parameters={"count": 2, "value": "100nF"},
            ports={"vdd": "3V3", "gnd": "GND"},
        ),
    ]
    for requirement in requirements:
        artifact = lower_requirement(requirement)
        assert artifact is not None, requirement.family
        for group in artifact.groups:
            lookup_pins(group.symbol, project_root=Path("."))
            lookup_footprint(group.footprint, project_root=Path("."))


def test_led_lowerer_forward_bias_and_current_follow_physical_symbol():
    from kicraft.design.synthesis.symbol_pinout import lookup_pins
    from kicraft.design.synthesis.validation import _resistance_ohms

    artifact = lower_requirement(
        _requirement(
            "led-current-resistor",
            parameters={"rail_voltage": 3.3, "led_vf": 2.0, "target_current_ma": 5},
            ports={"drive": "LED_DRIVE", "gnd": "GND"},
        )
    )
    led = next(group for group in artifact.groups if group.role == "led")
    names = {pin["number"]: pin["name"] for pin in lookup_pins(led.symbol)["pins"]}
    led_nets = {names[pin.pin]: pin.net for pin in artifact.pins if pin.role == "led"}
    resistor_nets = {pin.pin: pin.net for pin in artifact.pins if pin.role == "resistor"}
    assert led_nets["K"] == "GND"
    assert led_nets["A"] == resistor_nets["2"]
    assert resistor_nets["1"] == "LED_DRIVE"
    resistance = _resistance_ohms(
        next(group.value for group in artifact.groups if group.role == "resistor")
    )
    assert 0 < (3.3 - 2.0) / resistance <= 0.005
