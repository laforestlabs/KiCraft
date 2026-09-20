import json
import math

import pytest

from kicraft.design.lowering import lower_requirement, lowerer_contract_diagnostic
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
    with pytest.raises(ValueError):
        lower_requirement(requirement)


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
    with pytest.raises(ValueError):
        lower_requirement(_requirement("usb-c-breakout", ports=ports))


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
    with pytest.raises(ValueError):
        lower_requirement(requirement)


def test_header_physical_numbers_survive_sorted_json_round_trip():
    requirement = _requirement(
        "pin-header", ports={f"pin{index}": f"NET{index}" for index in range(1, 17)}
    )
    reordered = json.loads(json.dumps(requirement.model_dump(mode="json"), sort_keys=True))
    expected = {("connector", str(index)): f"NET{index}" for index in range(1, 17)}
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
    with pytest.raises(ValueError):
        lower_requirement(_requirement("pin-header", ports=ports))


@pytest.mark.parametrize(
    "updates",
    [
        {"parameters": {"pin_count": 20}},
        {"parameters": {"pitch_mm": 1.27}},
        {"exact_part": "TSW-118-07-L-D"},
    ],
)
def test_header_physical_constraints_reject_unproven_known_lowerer_claims(updates):
    requirement = _requirement("pin-header", ports={"pin1": "SIGNAL", "pin2": "GND"}).model_copy(
        update=updates
    )
    with pytest.raises(ValueError):
        lower_requirement(requirement)
    unit = StageWorkUnit(stage="bom", unit_id="bom:MAIN", sheet="MAIN", requirement_ids=("block",))
    state = _state(requirement)
    state["architecture"]["sheets"][0]["function"] = "2-pin header"
    with pytest.raises(ValueError):
        deterministic_bom_candidate(unit, state)


def test_screw_terminal_honors_registered_row_parameter():
    requirement = _requirement(
        "screw-terminal", parameters={"rows": 1}, ports={"positive": "VIN", "negative": "GND"}
    )
    artifact = lower_requirement(requirement)
    assert artifact is not None
    assert artifact.groups[0].symbol == "Connector:Screw_Terminal_01x02"
    with pytest.raises(ValueError):
        lower_requirement(requirement.model_copy(update={"parameters": {"rows": 2}}))


def test_selector_cannot_silently_substitute_an_explicit_device():
    requirement = _requirement(
        "voltage_selector_switch",
        parameters={"positions": 3},
        ports={"common": "CFG", "throw_1": "R9", "throw_2": "R12", "gnd": "RETURN"},
    )
    artifact = lower_requirement(requirement.model_copy(update={"exact_part": "SS13D07VG4"}))
    assert artifact.groups[0].mpn == "SS13D07VG4"
    with pytest.raises(ValueError):
        lower_requirement(requirement.model_copy(update={"exact_part": "MSK13C02-SZ"}))


def test_switch_lowerer_does_not_guess_an_absent_pull_policy():
    requirement = _requirement(
        "switch-input",
        parameters={"active_level": "high", "resistance": 10000},
        ports={"signal": "BOOT0", "gnd": "GND", "vdd": "+3V3"},
    )
    with pytest.raises(ValueError):
        lower_requirement(requirement)


def test_internal_switch_bias_cannot_discard_requested_resistor():
    requirement = _requirement(
        "switch-input",
        parameters={"pull_policy": "internal", "resistance": "4.7k"},
        ports={"signal": "BUTTON", "gnd": "GND", "vdd": "3V3"},
    )
    with pytest.raises(ValueError):
        lower_requirement(requirement)


def test_r2r_conflicting_parameter_aliases_reject_known_lowerer_claims():
    requirement = _requirement(
        "r2r-ladder",
        parameters={"bits": 2, "r_value": "10k", "r_series": "22k", "two_r_value": "20k"},
        ports={"bit0": "D0", "bit1": "D1", "output": "OUT", "gnd": "GND"},
    )
    with pytest.raises(ValueError):
        lower_requirement(requirement)


def test_selector_realizes_common_throws_grounded_frame_and_open_position():
    import json

    ports = json.loads(json.dumps(
        {"common": "CFG", "throw_1": "R9", "throw_2": "R12", "gnd": "FIELD_RETURN"},
        sort_keys=True,
    ))
    artifact = lower_requirement(
        _requirement("voltage_selector_switch", parameters={"positions": 3}, ports=ports)
    )
    assert {pin.pin: pin.net for pin in artifact.pins} == {
        "1": "R9", "2": "CFG", "3": "R12", "5": "FIELD_RETURN", "6": "FIELD_RETURN",
    }
    assert {pin.pin for pin in artifact.no_connects} == {"4"}
    all_throws = lower_requirement(
        _requirement(
            "voltage_selector_switch", parameters={"positions": 3},
            ports={**ports, "throw_3": "R20"},
        )
    )
    assert {pin.pin: pin.net for pin in all_throws.pins} == {
        "1": "R9", "2": "CFG", "3": "R12", "4": "R20",
        "5": "FIELD_RETURN", "6": "FIELD_RETURN",
    }
    assert not all_throws.no_connects


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
    with pytest.raises(ValueError):
        lower_requirement(_requirement("switch-input", parameters=parameters, ports=ports))


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
    with pytest.raises(ValueError):
        lower_requirement(requirement.model_copy(update=updates))

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
    with pytest.raises(ValueError):
        lower_requirement(requirement)


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
    with pytest.raises(ValueError):
        lower_requirement(requirement)


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
        _requirement("screw-terminal", ports={"p1": "VIN", "p2": "GND"}),
        _requirement(
            "fpc-connector",
            parameters={"pitch_mm": 0.5},
            ports={f"pin{index}": f"N{index}" for index in range(1, 25)},
        ),
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


def test_reviewed_fpc_connector_owns_only_the_24_contact_fpc():
    requirement = _requirement(
        "fpc-connector",
        parameters={"pitch_mm": 0.5},
        ports={f"pin{index}": f"FPC_{index}" for index in range(1, 25)},
    )

    artifact = lower_requirement(requirement)

    assert artifact is not None
    assert artifact.lowerer_id == "fpc-connector@1"
    assert [(group.role, group.mpn) for group in artifact.groups] == [
        ("fpc", "KH-FG0.5-H2.0-24PIN")
    ]
    assert {(pin.role, pin.pin) for pin in artifact.pins} == {
        ("fpc", str(index)) for index in range(1, 25)
    }




def test_reviewed_two_contact_terminal_preserves_exact_identity_and_polarity():
    artifact = lower_requirement(
        _requirement(
            "screw-terminal",
            parameters={"rows": 1},
            ports={"positive": "VOUT", "negative": "RETURN"},
        ).model_copy(update={"exact_part": "WJ126V-5.0-02P-14-00A"})
    )
    assert artifact.groups[0].mpn == "WJ126V-5.0-02P-14-00A"
    assert {pin.pin: pin.net for pin in artifact.pins} == {"1": "VOUT", "2": "RETURN"}
    assert not artifact.no_connects


def test_reviewed_three_position_terminal_uses_the_declared_three_rail_pin_order():
    requirement = _requirement(
        "screw-terminal",
        ports={"positive": "+12V", "common": "ISO_COM", "negative": "-12V"},
    ).model_copy(update={"exact_part": "WJ126V-5.0-03P-14-00A"})
    artifact = lower_requirement(
        json.loads(json.dumps(requirement.model_dump(mode="json"), sort_keys=True))
    )

    assert artifact is not None
    assert artifact.groups[0].mpn == "WJ126V-5.0-03P-14-00A"
    assert [(pin.pin, pin.net) for pin in artifact.pins] == [
        ("1", "+12V"),
        ("2", "ISO_COM"),
        ("3", "-12V"),
    ]


def test_reviewed_terminal_identity_resolves_in_its_canonical_lower_case_spelling():
    """The reviewed library's own spelling of a part must realize it.

    `part_identity` keys a screw terminal by its canonical identity
    (`wj126v-5.0-02p-14-00a`) while the shipped ordering code is the same string in
    another case. A private code table matched only the upper-case spelling, so the
    architecture derivation -- which writes the canonical identity onto the requirement --
    refused the very part it had just chosen (the 2026-09-18 self-eval: 29 refusals over
    10 briefs, the largest single fatal cause).
    """
    for spelling in ("wj126v-5.0-02p-14-00a", "WJ126V-5.0-02P-14-00A"):
        artifact = lower_requirement(
            _requirement(
                "screw-terminal",
                parameters={"rows": 1},
                ports={"pin1": "VOUT", "pin2": "RETURN"},
            ).model_copy(update={"exact_part": spelling})
        )
        assert artifact is not None, spelling
        # The claimed part is the shipped ordering code, never the raw draft spelling.
        assert artifact.groups[0].mpn == "WJ126V-5.0-02P-14-00A"
        assert artifact.groups[0].footprint == "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P"


def test_reviewed_four_contact_terminal_realizes_the_reviewed_four_way_part():
    """Every reviewed contact count is reachable, not only the two a table held."""
    artifact = lower_requirement(
        _requirement(
            "screw-terminal",
            parameters={"rows": 1},
            ports={f"pin{index}": f"NET{index}" for index in range(1, 5)},
        ).model_copy(update={"exact_part": "wj126v-5.0-04p-14-00a"})
    )

    assert artifact is not None
    assert artifact.groups[0].mpn == "WJ126V-5.0-04P-14-00A"
    assert artifact.groups[0].footprint == "screw-terminal-5mm-4p:CONN-TH_4P-P5.00_WJ126V-5.0-4P-1"
    assert len(artifact.pins) == 4


def test_unreviewed_terminal_identity_is_still_refused():
    with pytest.raises(ValueError):
        lower_requirement(
            _requirement(
                "screw-terminal",
                parameters={"rows": 1},
                ports={"pin1": "VOUT", "pin2": "RETURN"},
            ).model_copy(update={"exact_part": "WJ126V-5.0-02P"})
        )




def test_reviewed_bnc_connector_preserves_signal_and_all_shell_ground_pins():
    artifact = lower_requirement(
        _requirement(
            "bnc-connector",
            ports={"signal": "FILTER_OUT", "gnd": "GND"},
        ).model_copy(update={"exact_part": "KH-BNC50-3511"})
    )

    assert artifact is not None
    assert artifact.groups[0].mpn == "KH-BNC50-3511"
    assert {(pin.pin, pin.net) for pin in artifact.pins} == {
        ("1", "FILTER_OUT"),
        ("2", "GND"),
        ("3", "GND"),
        ("4", "GND"),
    }


def test_reviewed_trim_pot_rc_filter_has_an_adjustable_physical_resistance_path():
    artifact = lower_requirement(
        _requirement(
            "adjustable-rc-lowpass",
            parameters={
                "capacitance_f": 10e-9,
                "capacitor_exact_part": "C0805C103J5GACTU",
            },
            ports={"input": "BNC_IN", "output": "FILTER_OUT", "gnd": "GND"},
        ).model_copy(update={"exact_part": "3296W-1-103LF"})
    )

    assert artifact is not None
    assert [(group.role, group.mpn) for group in artifact.groups] == [
        ("trim_pot", "3296W-1-103LF"),
        ("capacitor", "C0805C103J5GACTU"),
    ]
    assert {
        (pin.role, pin.pin, pin.net) for pin in artifact.pins if pin.role == "trim_pot"
    } == {
        ("trim_pot", "1", "BNC_IN"),
        ("trim_pot", "2", "FILTER_OUT"),
        ("trim_pot", "3", "FILTER_OUT"),
    }


def test_reviewed_audio_jack_uses_named_contacts_and_mono_ring_no_connect():
    artifact = lower_requirement(
        _requirement(
            "audio-jack",
            ports={"sleeve": "AGND", "tip": "CHANNEL_IN", "ring": "NC"},
        ).model_copy(update={"exact_part": "SJ1-3533NG"})
    )

    assert artifact is not None
    assert artifact.groups[0].mpn == "SJ1-3533NG"
    assert {(pin.pin, pin.net) for pin in artifact.pins} == {
        ("S", "AGND"),
        ("T", "CHANNEL_IN"),
    }
    assert artifact.no_connects[0].pin == "R"


def test_capacitive_touch_lowerer_emits_two_board_fabricated_electrodes():
    from pathlib import Path

    import pcbnew

    from kicraft.design.synthesis.footprint_library import load_footprint
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    artifact = lower_requirement(
        _requirement(
            "capacitive-touch-pad",
            parameters={
                "count": 2,
                "pins": "PA4,PA5",
                "no_copper_underlay": True,
            },
            ports={"touch1": "TOUCH1", "touch2": "TOUCH2"},
        )
    )

    assert artifact is not None
    group = artifact.groups[0]
    assert (group.quantity, group.assembly) == (2, False)
    assert {(pin.index, pin.pin, pin.net) for pin in artifact.pins} == {
        (0, "1", "TOUCH1"),
        (1, "1", "TOUCH2"),
    }
    symbol = lookup_pins(group.symbol, all_units=True)
    footprint, _ = load_footprint(pcbnew, *group.footprint.split(":"), project_root=Path("."))
    pad = next(iter(footprint.Pads()))
    assert {pin["number"] for pin in symbol["pins"]} == {"1"}
    assert [candidate.GetNumber() for candidate in footprint.Pads()] == ["1"]
    assert (pcbnew.ToMM(pad.GetSizeX()), pcbnew.ToMM(pad.GetSizeY())) == (12.0, 12.0)
    assert pad.GetLayerSet().Contains(pcbnew.F_Cu)
    assert not pad.GetLayerSet().Contains(pcbnew.F_Mask)
    assert not pad.GetLayerSet().Contains(pcbnew.B_Cu)


@pytest.mark.parametrize(
    ("parameters", "ports"),
    [
        (
            {"count": 0, "pins": "", "no_copper_underlay": True},
            {},
        ),
        (
            {
                "count": 501,
                "pins": ",".join(f"PA{index}" for index in range(1, 502)),
                "no_copper_underlay": True,
            },
            {f"touch{index}": f"T{index}" for index in range(1, 502)},
        ),
        (
            {"count": 2, "pins": "PA4,PA5", "no_copper_underlay": True},
            {"touch1": "T1", "touch2": "T1"},
        ),
        (
            {"count": 2, "pins": "PA4,PA5", "no_copper_underlay": False},
            {"touch1": "T1", "touch2": "T2"},
        ),
    ],
)
def test_capacitive_touch_lowerer_refuses_unsupported_geometry_or_underlay(
    parameters, ports
):
    with pytest.raises(ValueError):
        lower_requirement(_requirement("capacitive-touch-pad", parameters=parameters, ports=ports))


def test_lowerer_refusal_names_missing_ports():
    """A lowerer-owned family with no declared contacts must say so.

    `screw-terminal@1` owns the family, so a model-authored fallback is refused; the
    requirement here simply declared no ports. The refusal must name that, not the
    generic "cannot realize this port/parameter combination", which a draft cannot
    act on.
    """
    diagnostic = lowerer_contract_diagnostic(_requirement("screw-terminal"))
    assert diagnostic is not None
    assert "needs this requirement's ports declared" in diagnostic.message
    # With its contacts bound, the lowerer realizes it and there is no refusal.
    realized = _requirement("screw-terminal", ports={"pin1": "+3V3", "pin2": "GND"})
    assert lowerer_contract_diagnostic(realized) is None


def test_lowerer_summary_publishes_required_ports():
    """The model must be told which ports a lowerer needs.

    These lowerers privately demanded specific ports in their build code while
    publishing no port contract, so a draft could not discover them and the refusal
    named nothing. `extras.circuit_lowerers` carries `required_port_keys` whenever the
    lowerer sets it.
    """
    from kicraft.design.lowering import lowerer_summaries

    rows = {row["lowerer"]: row for row in lowerer_summaries()}
    for lowerer_id, wanted in {
        "rc-lowpass@1": ["input", "output", "gnd"],
        "adjustable-rc-lowpass@1": ["input", "output", "gnd"],
        "voltage-divider@1": ["input", "output", "gnd"],
        "i2c-pullups@1": ["vdd", "sda", "scl"],
        "led-current-resistor@1": ["drive", "gnd"],
        "open-drain-pullup@1": ["vdd", "signal"],
        "switch-input@1": ["signal", "gnd", "vdd"],
        "coin-cell-holder@1": ["positive", "negative"],
        "explicit-decoupling@1": ["vdd", "gnd"],
    }.items():
        assert rows[lowerer_id]["required_port_keys"] == wanted


def test_lowerer_refusal_publishes_the_declared_contacts_and_the_family_contract():
    """A refusal must name the contacts the draft declared and the shapes the family builds.

    The 2026-09-18 self-eval refused five terminals whose contact numbers were gapped or
    written in a second key alphabet, with only the generic "cannot realize this
    port/parameter combination" text: the draft could not see which of its own contacts
    the lowerer rejected, nor which contact spellings the family accepts.
    """
    diagnostic = lowerer_contract_diagnostic(
        _requirement("screw-terminal", parameters={"rows": 1}, ports={"pin1": "A", "pin3": "B"})
    )

    assert diagnostic is not None
    assert "requirement_ports=pin1,pin3" in diagnostic.evidence
    contract = [line for line in diagnostic.evidence if line.startswith("port_contract=")]
    assert contract and "positive/negative" in contract[0]


def test_lowerer_refusal_names_required_ports():
    """An unported lowerer family names the exact ports its build code wants.

    `rc-lowpass@1` was the top live blocker: a draft declared none of input/output/gnd
    and got "cannot realize this port/parameter combination", which names nothing to
    bind, so the repair rounds repeated it.
    """
    diagnostic = lowerer_contract_diagnostic(
        _requirement("rc-lowpass", parameters={"cutoff_hz": 1000, "resistance_ohm": 10_000})
    )
    assert diagnostic is not None
    assert "input, output, gnd" in diagnostic.message
    assert "required_ports=input,output,gnd" in diagnostic.evidence
    # With its contacts bound the lowerer realizes it, so there is no refusal.
    bound = _requirement(
        "rc-lowpass",
        parameters={"cutoff_hz": 1000, "resistance_ohm": 10_000},
        ports={"input": "SIG_IN", "output": "SIG_FILT", "gnd": "GND"},
    )
    assert lowerer_contract_diagnostic(bound) is None


def test_lowerer_summary_publishes_the_reviewed_part_and_parameter_choices():
    """What a lowerer can build must be discoverable before a draft is written.

    `extras.circuit_lowerers` is the model's only view of these contracts: an audio jack
    that realizes exactly one reviewed part, and a filter whose two parameters have one
    accepted value each, are unreachable by a draft that cannot read them.
    """
    from kicraft.design.lowering import lowerer_summaries

    rows = {row["lowerer"]: row for row in lowerer_summaries()}
    assert rows["audio-jack@1"]["required_exact_part"] == "SJ1-3533NG"
    assert rows["bnc-connector@1"]["reviewed_exact_part"] == "KH-BNC50-3511"
    assert rows["fpc-connector@1"]["reviewed_exact_part"] == "KH-FG0.5-H2.0-24PIN"
    assert rows["adjustable-rc-lowpass@1"]["parameter_choices"] == {
        "capacitance_f": [10e-9],
        "capacitor_exact_part": ["C0805C103J5GACTU"],
    }
    # A family that realizes no specific reviewed part publishes none.
    assert "required_exact_part" not in rows["pin-header@1"]
    assert "reviewed_exact_part" not in rows["pin-header@1"]


def test_lowerer_refusal_names_the_reviewed_part_and_the_accepted_parameter():
    """The refusal must name the accepted value, not only that none was accepted.

    The canary (2026-09-17, `audio-jack-buffer`, `rc-lowpass-bnc`) refused four audio jacks
    and two filters with "cannot realize this port/parameter combination": the draft had
    bound every contact, and the one wrong fact — the ordering code, or one parameter
    value — was not published anywhere the draft could read.
    """
    jack_ports = {"sleeve": "GND", "tip": "IN", "ring": "OUT"}
    wrong_part = CircuitRequirement(
        id="jack",
        sheet="MAIN",
        role="connector",
        family="audio-jack",
        exact_part="SJ1-3533",
        ports=jack_ports,
    )
    diagnostic = lowerer_contract_diagnostic(wrong_part)
    assert diagnostic is not None
    assert "SJ1-3533NG" in diagnostic.message
    assert "required_exact_part=SJ1-3533NG" in diagnostic.evidence

    wrong_value = _requirement(
        "adjustable-rc-lowpass",
        parameters={"capacitance_f": 2.2e-6, "capacitor_exact_part": "C0805C103J5GACTU"},
        ports={"input": "IN", "output": "OUT", "gnd": "GND"},
    )
    diagnostic = lowerer_contract_diagnostic(wrong_value)
    assert diagnostic is not None
    assert "capacitance_f" in diagnostic.message
    assert "1e-08" in diagnostic.message
    assert "parameter_choices=capacitance_f=[1e-08]" in ";".join(diagnostic.evidence)


# ---------- the prototyping pad field: a board feature, not a component ------
#
# A prototyping shield's defining feature is a bare 2.54 mm pad field. It owns no
# contact, so its requirement declares no ports; the lowerer builds the pads from
# stock KiCad, declares every one a no-connect (no provider involvement at all) and
# declares the grid its members must be placed on instead of leaving them to the
# annealing solver.

_PAD_SYMBOL = "prototyping-area:PrototypingPad"
_PAD_FOOTPRINT = "prototyping-area:PrototypingPad_1.5mm_Drill0.8mm"
_PAD_SHEET = "PROTOTYPING AREA"


def _prototyping_area_requirement(**parameters) -> CircuitRequirement:
    return CircuitRequirement(
        id="prototyping_area",
        sheet=_PAD_SHEET,
        role="user_io",
        family="prototyping-area",
        parameters=parameters or {"rows": 5, "cols": 5, "pitch_mm": 2.54},
    )


def _prototyping_area_state(requirement: CircuitRequirement) -> dict:
    """The architecture the derivation writes for a pad-field obligation."""
    return {
        "architecture": {
            "sheets": [
                {
                    "name": _PAD_SHEET,
                    "stem": "PROTOTYPING_AREA",
                    "function": "Bare pad field the user solders through-hole parts into.",
                }
            ],
            "requirements": [requirement.model_dump(mode="json")],
            "recipe_selections": [],
            "power_nets": ["GND"],
            "inter_sheet_nets": [],
        },
        "bom": {"parts": []},
    }


def _lower_prototyping_area_bom():
    """The deterministic BOM the pipeline commits for one pad-field requirement."""
    from kicraft.design.models import BOM
    from kicraft.server.stage_contracts import _normalize_stage_response

    requirement = _prototyping_area_requirement()
    state = _prototyping_area_state(requirement)
    unit = StageWorkUnit("bom-r000", "bom", _PAD_SHEET, requirement_ids=(requirement.id,))
    candidate = deterministic_bom_candidate(unit, state)
    assert candidate is not None, "a ports-less board feature must not reach the provider"
    validated = validate_unit_candidate(
        unit, {**candidate, "_trusted_deterministic_candidate": True}, state, {}
    )
    merged, _ref_to_unit, ref_to_lowering, trusted = merge_bom_units(
        (unit,), {unit.unit_id: validated}, state
    )
    normalized, _count = _normalize_stage_response(
        "bom", merged, {"architecture": state["architecture"], "_trusted_lowering_group_ids": trusted}
    )
    return state, BOM.model_validate(normalized), ref_to_lowering


def test_prototyping_area_lowerer_builds_a_board_fabricated_pad_field():
    artifact = lower_requirement(_prototyping_area_requirement())
    assert artifact.lowerer_id == "prototyping-area@1"
    assert len(artifact.groups) == 1
    field = artifact.groups[0]
    assert (field.role, field.reference_prefix, field.quantity) == ("pad_field", "PB", 25)
    assert (field.symbol, field.footprint) == (_PAD_SYMBOL, _PAD_FOOTPRINT)
    assert field.assembly is False
    assert "2.54 mm" in field.value
    # The grid is the deliverable, so it travels with the group.
    assert field.array is not None
    assert (field.array.pattern, field.array.rows, field.array.cols) == ("grid", 5, 5)
    assert field.array.pitch_mm == 2.54
    # No pad carries a net: each is a declared no-connect at its own array index.
    assert artifact.pins == ()
    assert [(row.role, row.index, row.pin) for row in artifact.no_connects] == [
        ("pad_field", index, "1") for index in range(25)
    ]


def test_prototyping_area_lowerer_reads_the_declared_field_size():
    artifact = lower_requirement(_prototyping_area_requirement(rows=4, cols=8))
    field = artifact.groups[0]
    assert field.quantity == 32
    assert (field.array.rows, field.array.cols) == (4, 8)
    assert [row.index for row in artifact.no_connects] == list(range(32))


@pytest.mark.parametrize(
    "parameters",
    [
        {"rows": 4, "cols": 4, "pitch_mm": 2.54},  # 16 pads: no 0.1 inch part fits
        {"rows": 5, "cols": 5, "pitch_mm": 1.27},  # not the 0.1 inch grid
        {"rows": 25, "cols": 25, "pitch_mm": 2.54},  # 625 pads: beyond one group
        {"rows": "5", "cols": 5, "pitch_mm": 2.54},
        {"rows": True, "cols": 25, "pitch_mm": 2.54},
        {"rows": 5, "cols": 5, "pitch_mm": "2.54"},
    ],
)
def test_prototyping_area_lowerer_refuses_geometry_it_cannot_deliver(parameters):
    with pytest.raises(ValueError):
        lower_requirement(_prototyping_area_requirement(**parameters))


def test_prototyping_area_pad_symbol_pin_maps_to_its_footprint_pad():
    """§9.27: the symbol's one pin number must be a pad number on the footprint."""
    from pathlib import Path

    from kicraft.design.cli_app import _footprint_pad_numbers
    from kicraft.design.synthesis.footprint_library import lookup_footprint
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    pins = lookup_pins(_PAD_SYMBOL, all_units=True)["pins"]
    assert [(pin["number"], pin["electrical_type"]) for pin in pins] == [("1", "passive")]
    assert _footprint_pad_numbers(_PAD_FOOTPRINT, Path(".")) == {"1"}
    assert lookup_footprint(_PAD_FOOTPRINT, project_root=Path(".")) is not None


def test_prototyping_area_unit_passes_the_bom_identity_and_sourcing_gates():
    """The pads are bare copper: §9.27 resolves and §9.26 has nothing to order."""
    from pathlib import Path

    requirement = _prototyping_area_requirement()
    state = _prototyping_area_state(requirement)
    unit = StageWorkUnit("bom-r000", "bom", _PAD_SHEET, requirement_ids=(requirement.id,))
    candidate = deterministic_bom_candidate(unit, state)
    # A project root turns on the unit's own identity (§9.27 symbol/footprint pair,
    # resolvable library ids) and sourcing (§9.26) seams.
    validated = validate_unit_candidate(
        unit,
        {**candidate, "_trusted_deterministic_candidate": True},
        state,
        {"_validation_project_root": Path(".")},
    )
    assert [group["reference_prefix"] for group in validated["groups"]] == ["PB"]


def test_ports_less_requirement_without_a_builder_stays_model_owned():
    """The empty-ports rule still holds for a lowerer that needs declared contacts."""
    requirement = _requirement("pin-header", parameters={"rows": 1})
    state = _state(requirement)
    unit = StageWorkUnit("bom-r000", "bom", "MAIN", requirement_ids=(requirement.id,))
    assert deterministic_bom_candidate(unit, state) is None


def test_prototyping_area_bom_carries_its_grid_over_the_pad_refs():
    from kicraft.design.models import BOM

    state, bom, ref_to_lowering = _lower_prototyping_area_bom()
    assert [part.ref for part in bom.parts] == [f"PB{n}" for n in range(1, 26)]
    assert {part.assembly for part in bom.parts} == {False}
    assert len(bom.arrays) == 1
    grid = bom.arrays[0]
    assert grid.refs == [part.ref for part in bom.parts]
    assert (grid.pattern, grid.rows, grid.cols) == ("grid", 5, 5)
    assert grid.rows * grid.cols == len(grid.refs) == 25
    assert grid.pitch_mm == 2.54
    assert ref_to_lowering["PB1"] == {
        "resolution_id": "prototyping-area@1",
        "lowering_requirement_id": "prototyping_area",
        "lowering_role": "pad_field",
        "lowering_index": 0,
    }
    # Additive: a requirement that declares no pattern keeps the empty array list.
    header = _requirement("pin-header", parameters={"rows": 1}, ports={"pin1": "GND"})
    header_state = _state(header)
    header_unit = StageWorkUnit("bom-r000", "bom", "MAIN", requirement_ids=(header.id,))
    assert deterministic_bom_candidate(header_unit, header_state)["arrays"] == []


def test_prototyping_area_bom_keeps_a_model_declared_array_for_the_group():
    """One array per group: a pattern already declared for it is not duplicated."""
    requirement = _prototyping_area_requirement()
    state = _prototyping_area_state(requirement)
    unit = StageWorkUnit("bom-r000", "bom", _PAD_SHEET, requirement_ids=(requirement.id,))
    validated = validate_unit_candidate(
        unit,
        {"groups": [], "arrays": [{"group_id": "pad_field", "rows": 1, "cols": 25}]},
        state,
        {},
    )
    assert [array["group_id"] for array in validated["arrays"]] == ["pad_field"]
    assert (validated["arrays"][0]["rows"], validated["arrays"][0]["cols"]) == (1, 25)
    assert len(validated["groups"]) == 1


def test_prototyping_area_pads_are_declared_no_connects_after_expansion():
    """Every pad is addressed by (role, index) -> PB{index + 1} and left unwired."""
    from kicraft.design.models import PinEndpoint
    from kicraft.design.synthesis.validation import check_net_coverage

    state, bom, ref_to_lowering = _lower_prototyping_area_bom()
    state["bom"] = {
        "parts": [
            {
                **part.model_dump(exclude_none=True),
                "resolution_source": "lowerer",
                **ref_to_lowering[part.ref],
            }
            for part in bom.parts
        ]
    }
    extras = {
        "symbol_pinouts": {
            part.ref: {"pins": [{"number": "1", "electrical_type": "passive"}]}
            for part in bom.parts
        }
    }
    units = plan_stage_work_units("wiring", state, extras)
    assert len(units) == 1 and units[0].planned_resolution_source == "lowerer"
    wiring = deterministic_wiring_candidate(units[0], state, extras)
    assert wiring["pins"] == [
        {"ref": f"PB{n}", "pin": "1", "no_connect": True} for n in range(1, 26)
    ]
    # §9.11: every symbol pin is accounted for by the declared no-connects.
    committed = bom.model_copy(
        update={
            "no_connect_pins": [
                PinEndpoint(ref=row["ref"], pin=row["pin"]) for row in wiring["pins"]
            ]
        }
    )
    coverage = check_net_coverage(committed)
    assert coverage.ok, coverage.offenders


def test_prototyping_area_pads_are_what_populate_their_sheet():
    """§9.13: the derived sheet is a BOM work unit, not an empty sheet.

    A requirement-less sheet dies here -- it emits a blank leaf and any inter-sheet
    net routed through it has no pin to land on -- which is why the pad field's
    requirement and its parts are both derived rather than left to the model.
    """
    from kicraft.design.models import Architecture, BOM
    from kicraft.design.synthesis.validation import (
        check_bom_parts_reference_architecture_sheets,
        check_sheets_have_parts,
    )

    state, bom, _ref_to_lowering = _lower_prototyping_area_bom()
    architecture = Architecture.model_validate(state["architecture"])
    assert check_sheets_have_parts(architecture, bom).ok
    assert check_bom_parts_reference_architecture_sheets(architecture, bom).ok
    # The same sheet with no pads is exactly the empty sheet §9.13 refuses.
    assert not check_sheets_have_parts(architecture, BOM(parts=[])).ok
