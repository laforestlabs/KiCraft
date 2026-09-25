import json

from dataclasses import FrozenInstanceError

import pytest

from kicraft.server.stage_work_units import (
    StageDraftStore,
    StageWorkUnit,
    WorkUnitValidationError,
    deterministic_bom_candidate,
    deterministic_wiring_candidate,
    merge_bom_units,
    merge_wiring_units,
    plan_stage_work_units,
    route_work_unit_ids,
    stage_draft_fingerprint,
    validate_unit_candidate,
)
from kicraft.server.stage_contracts import BomComponentGroup


def test_bom_group_contract_rejects_unqualified_footprint():
    payload = _group("pd_controller", "A", prefix="U")
    payload["footprint"] = "QFN-32-1EP_5x5mm_P0.5mm_EP3.1x3.1mm"

    with pytest.raises(ValueError, match="footprint"):
        BomComponentGroup.model_validate(payload)


def _state(parts=None):
    return {
        "architecture": {
            "sheets": [{"name": "A"}, {"name": "B"}],
            "recipe_selections": [],
            "power_nets": [],
            "inter_sheet_nets": [],
        },
        "bom": {"parts": parts or []},
    }


def _pinout(count):
    return {"pins": [{"number": str(index)} for index in range(1, count + 1)]}


def _group(group_id, sheet, prefix="R", quantity=1):
    return {
        "id": group_id,
        "reference_prefix": prefix,
        "quantity": quantity,
        "value": "1k",
        "symbol": "Device:R",
        "footprint": "Resistor_SMD:R_0603_1608Metric",
        "sheet": sheet,
    }


def test_bom_unit_resolves_curated_bundle_from_exact_mpn():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    payload = {
        "groups": [
            {
                **_group("thermocouple_converter", "A", prefix="U"),
                "value": "MAX31855KASA+T",
                "mpn": "MAX31855KASA+T",
                "symbol": "Maxim_IC:MAX31855",
                "footprint": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
            }
        ]
    }

    validated = validate_unit_candidate(unit, payload, _state(), {})

    assert validated["groups"][0]["symbol"] == "max31855:MAX31855KASA+T"
    assert validated["groups"][0]["footprint"] == ("max31855:SO-8_L4.9-W3.9-P1.27-LS5.9-BL")


def test_bom_unit_resolves_curated_bundle_from_prefixed_mpn():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    payload = {
        "groups": [
            {
                **_group("trim_pot", "A", prefix="RV"),
                "value": "100k trimmer",
                "mpn": "Bourns 3296W-1-103LF",
                "symbol": "Potentiometer:Potentiometer",
                "footprint": "Potentiometer_SMD:Potentiometer_Bourns_3314J_Vertical",
            }
        ]
    }

    validated = validate_unit_candidate(unit, payload, _state(), {})

    assert validated["groups"][0]["symbol"] == "trim-pot-3296w-10k:3296W-1-103LF"
    assert validated["groups"][0]["footprint"] == "trim-pot-3296w-10k:RES-ADJ-TH_3296W"


def test_bom_unit_normalizes_known_legacy_potentiometer_symbol():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    payload = {
        "groups": [
            {
                **_group("trim_pot", "A", prefix="RV"),
                "mpn": "Bourns 3314J-1-104E",
                "symbol": "Potentiometer:Potentiometer_Bourns_3314J",
            }
        ]
    }

    validated = validate_unit_candidate(unit, payload, _state(), {})

    assert validated["groups"][0]["symbol"] == "trim-pot-3296w-10k:3296W-1-103LF"
    assert validated["groups"][0]["footprint"] == "trim-pot-3296w-10k:RES-ADJ-TH_3296W"
    assert validated["groups"][0]["mpn"] == "3296W-1-103LF"


def test_bom_unit_normalizes_invented_standard_connector_libraries():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    payload = {
        "groups": [
            {
                **_group("fpc_connector", "A", prefix="J"),
                "value": "FFC_CONNECTOR_24P_0.5MM",
                "symbol": "FPC_24PIN_0.5MM_SMT:FFC_CONNECTOR_24P_0.5MM",
                "footprint": "FPC_24PIN_0.5MM_SMT:FFC_CONNECTOR_24P_0.5MM",
            },
            {
                **_group("header_1x24", "A", prefix="J"),
                "value": "HEADER_1X24_2.54MM",
                "symbol": "HEADER_1X24_2.54MM:HEADER_1X24_2.54MM",
                "footprint": "HEADER_1X24_2.54MM:HEADER_1X24_2.54MM",
            },
        ]
    }

    validated = validate_unit_candidate(unit, payload, _state(), {})

    assert [(group["symbol"], group["footprint"]) for group in validated["groups"]] == [
        (
            "Connector_Generic:Conn_01x24",
            "Connector_FFC-FPC:Hirose_FH12-24S-0.5SH_1x24-1MP_P0.50mm_Horizontal",
        ),
        (
            "Connector_Generic:Conn_01x24",
            "Connector_PinHeader_2.54mm:PinHeader_1x24_P2.54mm_Vertical",
        ),
    ]


def test_empty_standard_connector_sheets_lower_deterministically():
    state = _state()
    state["architecture"]["sheets"] = [
        {
            "name": "FPC CONNECTOR",
            "function": "24-pin 0.5mm-pitch FPC/FFC connector for external cable",
        },
        {
            "name": "HEADER",
            "function": "0.1-inch header row exposing all 24 signals",
        },
    ]

    fpc = validate_unit_candidate(
        StageWorkUnit("bom-s000", "bom", "FPC CONNECTOR"),
        {"groups": []},
        state,
        {},
    )
    header = validate_unit_candidate(
        StageWorkUnit("bom-s001", "bom", "HEADER"),
        {"groups": []},
        state,
        {},
    )

    assert fpc["groups"][0]["footprint"] == (
        "Connector_FFC-FPC:Hirose_FH12-24S-0.5SH_1x24-1MP_P0.50mm_Horizontal"
    )
    assert header["groups"][0]["footprint"] == (
        "Connector_PinHeader_2.54mm:PinHeader_1x24_P2.54mm_Vertical"
    )


@pytest.mark.parametrize(
    ("function", "expected_ids"),
    [
        (
            "USB-C receptacle with CC pull-downs, providing 5V VBUS and GND",
            {"connector", "cc_pulldown"},
        ),
        (
            "USB-C connector with ESD protection and VBUS detection",
            {"connector", "cc_pulldown", "esd", "usb_series"},
        ),
    ],
)
def test_empty_usb_c_sheets_lower_to_standard_recipe_parts(function, expected_ids):
    state = _state()
    state["architecture"]["sheets"] = [{"name": "USB INPUT", "function": function}]

    candidate = validate_unit_candidate(
        StageWorkUnit("bom-s000", "bom", "USB INPUT"),
        {"groups": []},
        state,
        {},
    )

    assert {group["id"] for group in candidate["groups"]} == expected_ids
    assert all(group["sheet"] == "USB INPUT" for group in candidate["groups"])


def test_usb_signal_header_is_not_replaced_with_a_receptacle():
    state = _state()
    state["architecture"]["sheets"] = [
        {
            "name": "HEADER BREAKOUT",
            "function": "24-pin header exposing USB-C connector signals and VBUS",
        }
    ]

    candidate = validate_unit_candidate(
        StageWorkUnit("bom-s000", "bom", "HEADER BREAKOUT"),
        {"groups": []},
        state,
        {},
    )

    assert [(group["symbol"], group["footprint"]) for group in candidate["groups"]] == [
        (
            "Connector_Generic:Conn_01x24",
            "Connector_PinHeader_2.54mm:PinHeader_1x24_P2.54mm_Vertical",
        )
    ]


def test_standard_power_terminal_resolves_installed_identity(tmp_path):
    from kicraft.server.stage_work_units import _bom_unit_identity_defects

    state = _state()
    state["architecture"]["sheets"] = [
        {"name": "POWER INPUT", "function": "Two-pin power input terminal"}
    ]
    candidate = deterministic_bom_candidate(StageWorkUnit("bom-s000", "bom", "POWER INPUT"), state)
    groups = [BomComponentGroup.model_validate(group) for group in candidate["groups"]]

    assert _bom_unit_identity_defects(groups, tmp_path) == {
        "unresolved-footprint": [],
        "unresolved-symbol": [],
        "symbol-footprint-pad-mismatch": [],
    }


def test_selectable_usb_pd_trigger_stays_model_owned():
    state = _state()
    state["architecture"]["sheets"] = [
        {
            "name": "PD",
            "function": "USB-C PD trigger with switch-selectable output voltage",
        }
    ]

    candidate = deterministic_bom_candidate(
        StageWorkUnit("bom-s000", "bom", "PD"),
        state,
    )

    assert candidate is None


def test_typed_pd_power_input_is_not_captured_by_connector_sheet_lowerer():
    state = _state()
    state["architecture"]["sheets"] = [
        {"name": "PD", "function": "USB-C receptacle and power interface"}
    ]
    state["architecture"]["requirements"] = [
        {
            "id": "power_controller",
            "sheet": "PD",
            "role": "power_input",
            "family": "usb-pd-trigger",
            "parameters": {"supported_voltages": "9V,12V,20V"},
            "ports": {"cc1": "CC1", "cc2": "CC2", "vbus_out": "VBUS"},
        }
    ]

    (unit,) = plan_stage_work_units("bom", state, {})

    assert unit.requirement_ids == ("power_controller",)
    assert deterministic_bom_candidate(unit, state) is None


def test_mcu_sheet_cannot_be_replaced_by_ancillary_ldo():
    state = _state()
    state["architecture"]["sheets"] = [
        {
            "name": "CORE",
            "function": "Module with integrated MCU, USB transceiver, and 3.3V LDO regulator.",
        }
    ]

    (unit,) = plan_stage_work_units("bom", state, {})

    assert deterministic_bom_candidate(unit, state) is None


def test_typed_mcu_cannot_be_replaced_by_ancillary_ldo_without_prose_hint():
    from kicraft.design.models import CircuitRequirement

    state = _state()
    state["architecture"]["sheets"] = [
        {"name": "CORE", "function": "Processing module with integrated 3.3V LDO regulator"}
    ]
    state["architecture"]["requirements"] = [
        CircuitRequirement(
            id="processing", sheet="CORE", role="mcu_core", family="custom-processing-module"
        ).model_dump(mode="json")
    ]
    (unit,) = plan_stage_work_units("bom", state, {})

    assert deterministic_bom_candidate(unit, state) is None
    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, {"groups": [_group("ldo_caps", "CORE")]}, state, {})
    assert caught.value.defects["missing-requirement-implementation"] == ["processing"]


@pytest.mark.parametrize(
    ("function", "expected_id"),
    [
        ("Per-port current limiting power switch with overcurrent flag", "current_limit_switch"),
        ("Status LEDs for power present and overcurrent indication", "status_led"),
        ("Two-pin external +3V3 and GND power input", "power_input"),
    ],
)
def test_common_standalone_sheets_lower_deterministically(function, expected_id):
    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": function}]

    candidate = validate_unit_candidate(
        StageWorkUnit("bom-s000", "bom", "A"),
        {"groups": []},
        state,
        {},
    )

    assert expected_id in {group["id"] for group in candidate["groups"]}


def test_standard_connectors_wire_from_ordered_sheet_interfaces():
    state = _state([{"ref": "J1", "sheet": "A", "symbol": "Connector_Generic:Conn_01x03"}])
    state["architecture"]["inter_sheet_nets"] = [
        {
            "name": name,
            "endpoints": [
                {"sheet": "A", "direction": "passive"},
                {"sheet": "B", "direction": "passive"},
            ],
        }
        for name in ("SIG1", "SIG2", "SIG3")
    ]
    state["bom"]["parts"][0]["resolution_source"] = "lowerer"
    unit = StageWorkUnit(
        "wiring-u000",
        "wiring",
        "A",
        ("J1",),
        (("J1", "1"), ("J1", "2"), ("J1", "3")),
    )

    candidate = deterministic_wiring_candidate(
        unit,
        state,
        {"symbol_pinouts": {"J1": _pinout(3)}},
    )

    assert [row["net"] for row in candidate["pins"]] == ["SIG1", "SIG2", "SIG3"]


def test_bnc_wiring_grounds_every_shell_pin():
    state = _state([{"ref": "J1", "sheet": "A", "symbol": "bnc-pcb-jack:KH-BNC50-3511"}])
    state["architecture"]["power_nets"] = ["GND"]
    state["architecture"]["inter_sheet_nets"] = [
        {
            "name": name,
            "endpoints": [
                {"sheet": "A", "direction": "passive"},
                {"sheet": "B", "direction": "passive"},
            ],
        }
        for name in ("SIG", "GND")
    ]
    state["bom"]["parts"][0]["resolution_source"] = "lowerer"
    unit = StageWorkUnit(
        "wiring-u000",
        "wiring",
        "A",
        ("J1",),
        tuple(("J1", str(pin)) for pin in range(1, 5)),
    )

    candidate = deterministic_wiring_candidate(
        unit,
        state,
        {"symbol_pinouts": {"J1": _pinout(4)}},
    )

    assert [row["net"] for row in candidate["pins"]] == ["SIG", "GND", "GND", "GND"]


def test_combined_fpc_header_sheet_wires_matching_internal_signals():
    state = _state([{"ref": "J1", "sheet": "A", "symbol": "Connector_Generic:Conn_01x03"}])
    state["architecture"]["sheets"] = [{"name": "A", "function": "FPC breakout to matching header"}]
    state["bom"]["parts"][0]["resolution_source"] = "lowerer"
    unit = StageWorkUnit(
        "wiring-u000",
        "wiring",
        "A",
        ("J1",),
        (("J1", "1"), ("J1", "2"), ("J1", "3")),
    )

    candidate = deterministic_wiring_candidate(
        unit,
        state,
        {"symbol_pinouts": {"J1": _pinout(3)}},
    )

    assert [row["net"] for row in candidate["pins"]] == ["SIG1", "SIG2", "SIG3"]


def test_bom_connector_unit_discards_sibling_circuit_groups():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "input_connector",
            "sheet": "A",
            "role": "connector",
            "family": "bnc_connector",
        },
        {
            "id": "filter",
            "sheet": "A",
            "role": "analog_block",
            "family": "rc_filter",
        },
    ]
    unit = StageWorkUnit(
        "bom-r000",
        "bom",
        "A",
        requirement_ids=("input_connector",),
        owned_roles=("connector",),
    )
    payload = {
        "groups": [
            _group("input", "A", prefix="J"),
            _group("filter_resistor", "A"),
            _group("filter_capacitor", "A", prefix="C"),
        ]
    }

    validated = validate_unit_candidate(unit, payload, state, {})

    assert [group["id"] for group in validated["groups"]] == ["input"]


def test_bom_planning_keeps_explicit_power_requirements():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "external_power",
            "sheet": "A",
            "role": "power_input",
            "family": "power_input",
        },
        {
            "id": "filter",
            "sheet": "A",
            "role": "analog_block",
            "family": "rc_filter",
        },
    ]

    units = plan_stage_work_units("bom", state, {})

    # Every unselected requirement remains work, even without a typed lowerer.
    assert [(unit.requirement_ids, unit.sheet) for unit in units] == [
        (("external_power", "filter"), "A"),
        ((), "B"),
    ]


def test_units_are_immutable_and_bom_follows_architecture_order():
    units = plan_stage_work_units("bom", _state(), {})
    assert [(unit.unit_id, unit.sheet) for unit in units] == [
        ("bom-s000", "A"),
        ("bom-s001", "B"),
    ]
    with pytest.raises(FrozenInstanceError):
        units[0].sheet = "B"


def test_wiring_packs_complete_refs_at_256_pin_boundary():
    parts = [
        {"ref": "U1", "sheet": "A", "symbol": "X:U"},
        {"ref": "U2", "sheet": "A", "symbol": "X:U"},
        {"ref": "R1", "sheet": "A", "symbol": "X:R"},
    ]
    units = plan_stage_work_units(
        "wiring",
        _state(parts),
        {"symbol_pinouts": {"U1": _pinout(128), "U2": _pinout(128), "R1": _pinout(1)}},
    )
    assert [len(unit.expected_pins) for unit in units] == [256, 1]
    assert units[0].refs == ("U1", "U2")
    assert units[1].refs == ("R1",)


def test_wiring_slices_only_an_oversized_ref():
    parts = [
        {"ref": "U1", "sheet": "A", "symbol": "X:U"},
        {"ref": "R1", "sheet": "A", "symbol": "X:R"},
    ]
    units = plan_stage_work_units(
        "wiring",
        _state(parts),
        {"symbol_pinouts": {"U1": _pinout(300), "R1": _pinout(2)}},
    )
    assert [len(unit.expected_pins) for unit in units] == [256, 44, 2]
    assert units[0].refs == units[1].refs == ("U1",)
    assert units[2].refs == ("R1",)


def test_recipe_connected_and_no_connect_pins_are_excluded():
    parts = [{"ref": "U1", "sheet": "A", "symbol": "X:U"}]
    units = plan_stage_work_units(
        "wiring",
        _state(parts),
        {
            "symbol_pinouts": {"U1": _pinout(4)},
            "locked_pin_assignments": [{"ref": "U1", "pin": "1", "net": "GND"}],
            "locked_no_connect_pins": [{"ref": "U1", "pin": "4"}],
        },
    )
    assert units[0].expected_pins == (("U1", "2"), ("U1", "3"))


def test_bom_validation_aggregates_all_owned_defect_classes():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    group = _group("r", "B")
    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(
            unit,
            {"groups": [group, group], "arrays": [{"group_id": "missing"}]},
            _state(),
            {},
        )
    defects = caught.value.defects
    assert defects["wrong-sheet"] == ["r:B", "r:B"]
    assert defects["duplicate-group"] == ["r"]
    assert defects["bad-array-reference"] == ["missing"]


def test_bom_validation_rejects_an_unpopulated_nonrecipe_sheet():
    unit = StageWorkUnit("bom-s000", "bom", "A")

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, {"groups": [], "arrays": []}, _state(), {})

    assert caught.value.defects["empty-sheet"] == ["A"]


def test_wiring_validation_reports_exact_coverage_and_ownership_defects():
    state = _state([{"ref": "U1", "sheet": "A", "symbol": "X:U"}])
    extras = {
        "symbol_pinouts": {"U1": _pinout(3)},
        "locked_pin_assignments": [{"ref": "U1", "pin": "1", "net": "GND"}],
    }
    unit = StageWorkUnit("wiring-u000", "wiring", "A", ("U1",), (("U1", "1"), ("U1", "2")))
    payload = {
        "pins": [
            {"ref": "U1", "pin": "1", "net": "A"},
            {"ref": "U1", "pin": "1", "net": "B"},
            {"ref": "U1", "pin": "9", "net": "C"},
            {"ref": "X1", "pin": "1", "no_connect": True},
        ]
    }
    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, payload, state, extras)
    defects = caught.value.defects
    assert defects["duplicate"] == ["U1.1"]
    assert defects["missing"] == ["U1.2"]
    assert defects["unexpected"] == ["U1.9", "X1.1"]
    assert defects["unknown-ref"] == ["X1"]
    assert defects["unknown-pin"] == ["U1.9"]
    assert defects["recipe-owned"] == ["U1.1"]


def test_bom_merge_rewrites_ids_arrays_and_stably_deduplicates():
    state = _state()
    units = plan_stage_work_units("bom", state, {})
    candidates = {
        "bom-s000": {
            "groups": [_group("res", "A")],
            "arrays": [{"group_id": "res", "pattern": "grid", "rows": 1, "cols": 1}],
            "assumptions": ["shared (defaulted)"],
            "substitutions": [],
        },
        "bom-s001": {
            "groups": [_group("res", "B")],
            "arrays": [],
            "assumptions": ["shared (defaulted)"],
            "substitutions": [],
        },
    }
    merged, refs, lowering, trusted_groups = merge_bom_units(units, candidates, state)
    assert [group["id"] for group in merged["groups"]] == ["s000_res", "s001_res"]
    assert merged["arrays"][0]["group_id"] == "s000_res"
    assert merged["assumptions"] == ["shared (defaulted)"]
    assert trusted_groups == frozenset()
    assert refs == {"R1": "bom-s000", "R2": "bom-s001"}
    assert lowering == {}


def test_wiring_merge_orders_pins_and_routes_exact_offenders():
    units = (
        StageWorkUnit("wiring-u000", "wiring", "A", ("U1",), (("U1", "1"),)),
        StageWorkUnit("wiring-u001", "wiring", "B", ("U10",), (("U10", "1"),)),
    )
    candidates = {
        "wiring-u000": {"pins": [{"ref": "U1", "pin": "1", "net": "A"}]},
        "wiring-u001": {"pins": [{"ref": "U10", "pin": "1", "net": "B"}]},
    }

    merged, pins, refs = merge_wiring_units(units, candidates)
    assert [row["ref"] for row in merged["pins"]] == ["U1", "U10"]
    assert route_work_unit_ids(
        {"offenders": ["U10.1"]}, units, pin_to_unit=pins, ref_to_unit_ids=refs
    ) == ("wiring-u001",)
    assert route_work_unit_ids({"errors": ["unscoped"]}, units) == (
        "wiring-u000",
        "wiring-u001",
    )


def test_route_work_unit_ids_matches_sheet_case_insensitively():
    units = (
        StageWorkUnit("bom-s000", "bom", "AMPLIFIER INPUT"),
        StageWorkUnit("bom-s001", "bom", "HIGH PASS FILTER"),
    )
    # Semantic diagnostics lowercase evidence; the sheet name is schema-forced
    # uppercase. The owning unit must still be targeted, not the fallback set.
    evidence = [
        {"message": "no active amplifier on its input sheet", "evidence": ["amplifier input"]}
    ]
    assert route_work_unit_ids(evidence, units) == ("bom-s000",)


def test_exact_pin_offender_routes_only_its_slice_of_an_oversized_ref():
    units = (
        StageWorkUnit("wiring-u000", "wiring", "A", ("U1",), (("U1", "1"),)),
        StageWorkUnit("wiring-u001", "wiring", "A", ("U1",), (("U1", "299"),)),
    )
    candidates = {
        "wiring-u000": {"pins": [{"ref": "U1", "pin": "1", "net": "A"}]},
        "wiring-u001": {"pins": [{"ref": "U1", "pin": "299", "net": "B"}]},
    }
    _merged, pins, refs = merge_wiring_units(units, candidates)

    assert route_work_unit_ids(
        {"offenders": ["U1.299"]},
        units,
        pin_to_unit=pins,
        ref_to_unit_ids=refs,
    ) == ("wiring-u001",)


def test_overlapping_wiring_ownership_is_rejected():
    units = (
        StageWorkUnit("a", "wiring", "A", ("U1",), (("U1", "1"),)),
        StageWorkUnit("b", "wiring", "A", ("U1",), (("U1", "1"),)),
    )
    candidates = {
        unit.unit_id: {"pins": [{"ref": "U1", "pin": "1", "net": unit.unit_id}]} for unit in units
    }
    with pytest.raises(ValueError, match="overlapping wiring ownership"):
        merge_wiring_units(units, candidates)


def test_checkpoint_resume_invalidation_and_corruption(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    units = (StageWorkUnit("wiring-u000", "wiring", "A", ("U1",), (("U1", "1"),)),)
    candidate = {"wiring-u000": {"pins": [{"ref": "U1", "pin": "1", "net": "A"}]}}
    fingerprint = stage_draft_fingerprint(
        stage="wiring",
        brief="brief",
        prompt_state={"upstream": 1},
        answers=None,
        instruction=None,
        stage_spec_sha256="spec",
        response_contract_names=("v3",),
        units=units,
        extras={"pinouts": 1},
    )
    store = StageDraftStore(state_path, "wiring")
    store.save(fingerprint, units, candidate)
    assert store.load(fingerprint, units) == candidate
    assert store.load("changed", units) == {}
    store.path.write_text("{broken")
    assert store.load(fingerprint, units) == {}


def test_replacing_one_candidate_preserves_unaffected_unit():
    units = (
        StageWorkUnit("wiring-u000", "wiring", "A", ("U1",), (("U1", "1"),)),
        StageWorkUnit("wiring-u001", "wiring", "B", ("R1",), (("R1", "1"),)),
    )
    candidates = {
        "wiring-u000": {"pins": [{"ref": "U1", "pin": "1", "net": "OLD"}]},
        "wiring-u001": {"pins": [{"ref": "R1", "pin": "1", "net": "STABLE"}]},
    }
    first, _, _ = merge_wiring_units(units, candidates)
    candidates["wiring-u000"] = {"pins": [{"ref": "U1", "pin": "1", "net": "NEW"}]}
    second, _, _ = merge_wiring_units(units, candidates)
    assert (
        first["pins"][1]
        == second["pins"][1]
        == {
            "ref": "R1",
            "pin": "1",
            "net": "STABLE",
        }
    )


def test_recipe_complete_requirement_omits_bom_work_unit():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "mcu_core",
            "sheet": "A",
            "role": "mcu_core",
            "family": "esp32-s3-module",
        }
    ]
    state["architecture"]["recipe_selections"] = [
        {
            "recipe": "esp32-s3-mini-1-minimal@1",
            "instance": "mcu_core",
            "sheets": {"mcu": "A"},
            "requirement_ids": ["mcu_core"],
        }
    ]
    units = plan_stage_work_units("bom", state, {})
    assert all("mcu_core" not in unit.requirement_ids for unit in units)
    assert [unit.sheet for unit in units] == ["B"]


def test_recipe_part_role_does_not_own_an_unselected_requirement():
    state = _state()
    state["architecture"]["requirements"] = [
        {"id": "mcu", "sheet": "A", "role": "mcu_core", "family": "rp2040"},
        {"id": "xtal", "sheet": "A", "role": "analog_block", "family": "crystal"},
    ]
    state["architecture"]["recipe_selections"] = [
        {
            "recipe": "rp2040-minimal@2",
            "instance": "mcu",
            "sheets": {"mcu": "A", "io": "A"},
            "requirement_ids": ["mcu"],
        }
    ]

    units = plan_stage_work_units("bom", state, {})

    assert [unit.requirement_ids for unit in units if unit.sheet == "A"] == [("xtal",)]


def test_recipe_internal_role_id_does_not_own_an_unselected_requirement():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "mcu",
            "sheet": "A",
            "role": "mcu_core",
            "family": "stm32f103c8",
        },
        {
            "id": "hse_crystal",
            "sheet": "A",
            "role": "analog_block",
            "family": "crystal",
        },
    ]
    state["architecture"]["recipe_selections"] = [
        {
            "recipe": "stm32f103c8t6-minimal@1",
            "instance": "mcu",
            "sheets": {"mcu": "A"},
            "requirement_ids": ["mcu"],
        }
    ]

    units = plan_stage_work_units("bom", state, {})

    assert [unit.requirement_ids for unit in units if unit.sheet == "A"] == [("hse_crystal",)]


def test_mixed_recipe_sheet_plans_only_unresolved_role():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "mcu_core",
            "sheet": "A",
            "role": "mcu_core",
            "family": "esp32-s3-module",
        },
        {
            "id": "novel_analog",
            "sheet": "A",
            "role": "analog_block",
            "family": "application-specific",
        },
    ]
    state["architecture"]["recipe_selections"] = [
        {
            "recipe": "esp32-s3-mini-1-minimal@1",
            "instance": "mcu_core",
            "sheets": {"mcu": "A"},
            "requirement_ids": ["mcu_core"],
        }
    ]
    units = plan_stage_work_units("bom", state, {})
    # Sheet A: the recipe-owned mcu_core is omitted, its plain sibling rolls up
    # into a sheet unit. Sheet B: uncovered, gets its own sheet unit.
    assert len(units) == 2
    assert units[0].requirement_ids == ("novel_analog",)
    assert units[0].owned_roles == ("analog_block",)
    assert units[0].recipe_ids == ("esp32-s3-mini-1-minimal@1",)
    assert units[1].sheet == "B"


def test_bom_unit_rejects_protected_identity_before_merge():
    unit = StageWorkUnit(
        "bom-r000",
        "bom",
        "A",
        requirement_ids=("novel_analog",),
    )
    payload = {
        "groups": [
            {
                "id": "esp32_s3_support",
                "reference_prefix": "C",
                "quantity": 1,
                "value": "100nF",
                "symbol": "Device:C",
                "footprint": "Capacitor_SMD:C_0603_1608Metric",
                "sheet": "A",
            }
        ]
    }
    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, payload, _state(), {})
    assert caught.value.defects["model_authored_protected_identity"] == ["esp32_s3_support"]


def test_generic_power_input_cannot_claim_unselected_registered_regulator():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "power_input",
            "sheet": "A",
            "role": "power_input",
            "family": "power-input",
            "parameters": {"input_voltage": 5.0, "output_voltage": 3.3},
            "ports": {"gnd": "GND", "output": "+3V3"},
        }
    ]
    # The saved CAN candidate remains invalid even if an old persisted plan
    # reaches validation: registered identities are protected globally, not
    # merely when another selected recipe happens to own this component.
    unit = StageWorkUnit("bom-s002", "bom", "A", requirement_ids=("power_input",))
    regulator = {
        **_group("ldo_3v3", "A", prefix="U"),
        "value": "AMS1117-3.3",
        "mpn": "AMS1117-3.3",
        "symbol": "Regulator_Linear:AMS1117-3.3",
        "footprint": "Package_TO_SOT_SMD:SOT-223-3_TabPin2",
    }
    with pytest.raises(WorkUnitValidationError) as rejected:
        validate_unit_candidate(unit, {"groups": [regulator]}, state, {})
    assert rejected.value.defects["model_authored_protected_identity"] == ["ldo_3v3"]


def test_bom_unit_allows_owned_protected_identity():
    # A protected class is realized by reviewed hardware: the requirement names the
    # reviewed 16-pin receptacle identity and the group is that pair, so ownership is
    # physical rather than a matching group label.
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "usb_c_receptacle",
            "sheet": "A",
            "role": "connector",
            "family": "usb_c_receptacle",
            "exact_part": "TYPE-C-31-M-12",
        }
    ]
    unit = StageWorkUnit(
        "bom-r000",
        "bom",
        "A",
        requirement_ids=("usb_c_receptacle",),
        owned_roles=("connector",),
    )
    payload = {
        "groups": [
            {
                "id": "usb_c_receptacle",
                "reference_prefix": "J",
                "quantity": 1,
                "value": "TYPE-C-31-M-12",
                "mpn": "TYPE-C-31-M-12",
                "symbol": "usb-c-16p:TYPE-C-31-M-12",
                "footprint": "usb-c-16p:USB-C_SMD-TYPE-C-31-M-12_1",
                "sheet": "A",
            }
        ]
    }

    validated = validate_unit_candidate(unit, payload, state, {})

    assert validated["groups"][0]["id"] == "usb_c_receptacle"
    assert validated["groups"][0]["symbol"] == "usb-c-16p:TYPE-C-31-M-12"


def test_bom_unit_allows_owned_protected_identity_words():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "pd_trigger_controller",
            "sheet": "A",
            "role": "bus_interface",
            "family": "pd_trigger_controller",
            "exact_part": "CH224K",
        }
    ]
    unit = StageWorkUnit(
        "bom-r000",
        "bom",
        "A",
        requirement_ids=("pd_trigger_controller",),
        owned_roles=("bus_interface",),
    )
    payload = {
        "groups": [
            {
                "id": "pd_controller",
                "reference_prefix": "U",
                "quantity": 1,
                "value": "CH224K",
                "symbol": "ch224k:CH224K",
                "footprint": "ch224k:SOT-23-6_WRONG",
                "sheet": "A",
            }
        ]
    }

    validated = validate_unit_candidate(unit, payload, state, {})

    assert validated["groups"][0]["footprint"] == ("ch224k:ESSOP-10_L4.9-W3.9-P1.0-LS6.0-TL-EP")


def test_bom_unit_rejects_parts_that_do_not_implement_owned_requirement():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "pd_trigger_controller",
            "sheet": "A",
            "role": "bus_interface",
            "family": "pd_trigger_controller",
        }
    ]
    unit = StageWorkUnit(
        "bom-r000",
        "bom",
        "A",
        requirement_ids=("pd_trigger_controller",),
        owned_roles=("bus_interface",),
    )
    payload = {
        "groups": [
            {
                "id": "support_resistor",
                "reference_prefix": "R",
                "quantity": 1,
                "value": "10k",
                "symbol": "Device:R",
                "footprint": "Resistor_SMD:R_0603_1608Metric",
                "sheet": "A",
            }
        ]
    }

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, payload, state, {})

    assert caught.value.defects["missing-requirement-implementation"] == ["pd_trigger_controller"]


@pytest.mark.parametrize("family", ["switch-input", "voltage_selector_switch"])
@pytest.mark.parametrize("with_sibling", [False, True])
def test_typed_switch_requirement_rejects_unrelated_passives(family, with_sibling):
    state = _state()
    state["architecture"]["sheets"] = [
        {"name": "A", "function": "Control circuit with status LED and supply decoupling"}
    ]
    requirements = [{"id": "control", "sheet": "A", "role": "user_io", "family": family}]
    if with_sibling:
        requirements.append(
            {"id": "filter", "sheet": "A", "role": "analog_block", "family": "custom-filter"}
        )
    state["architecture"]["requirements"] = requirements
    (unit,) = plan_stage_work_units("bom", state, {})
    assert deterministic_bom_candidate(unit, state) is None
    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(
            unit,
            {
                "groups": [
                    _group("control", "A", prefix="SW"),
                    {
                        **_group("filter", "A", prefix="C"),
                        "symbol": "Device:C",
                        "value": "100nF",
                        "footprint": "Capacitor_SMD:C_0603_1608Metric",
                    },
                ],
                "_lowering_requirement_id": "control",
            },
            state,
            {},
        )
    assert caught.value.defects["missing-requirement-implementation"] == ["control"]


def test_button_requirement_accepts_physical_button_with_ancillary_passives():
    state = _state()
    state["architecture"]["requirements"] = [
        {"id": "control", "sheet": "A", "role": "user_io", "family": "switch-input"}
    ]
    unit = StageWorkUnit("bom-r000", "bom", "A", requirement_ids=("control",))
    switch = {
        **_group("control_switch", "A", prefix="SW"),
        "value": "BUTTON",
        "symbol": "Switch:SW_Push",
        "footprint": "Button_Switch_SMD:SW_SPST_TL3342",
    }
    validated = validate_unit_candidate(
        unit, {"groups": [switch, _group("pullup", "A")]}, state, {}
    )
    assert {group["id"] for group in validated["groups"]} == {"control_switch", "pullup"}


def test_typed_coin_cell_holder_replaces_a_welded_cell_footprint():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "battery",
            "sheet": "A",
            "role": "power_input",
            "family": "coin-cell-holder",
            "parameters": {"cell_format": "CR2032"},
            "ports": {"positive": "VBAT", "negative": "GND"},
        }
    ]
    unit = StageWorkUnit("bom-r000", "bom", "A", requirement_ids=("battery",))
    validated = validate_unit_candidate(
        unit,
        {
            "groups": [
                {
                    **_group("battery_holder", "A", prefix="BT"),
                    "value": "CR2032",
                    "symbol": "Device:Battery_Cell",
                    "footprint": "Battery:Battery_Panasonic_CR2032-HFN_Horizontal_CircularHoles",
                }
            ]
        },
        state,
        {},
    )
    # The typed lowerer owns this requirement: the model's welded-cell footprint
    # is discarded in favour of the real curated holder.
    holder = validated["groups"][0]
    assert holder["footprint"].startswith("Battery:BatteryHolder_")
    assert holder["mpn"] == "BS-07-A1BJ001"


def _battery_power_state():
    state = _state()
    state["architecture"]["sheets"] = [
        {"name": "A", "function": "Battery and voltage conditioning"}
    ]
    state["architecture"]["requirements"] = [
        {
            "id": "battery",
            "sheet": "A",
            "role": "power_input",
            "family": "coin-cell-holder",
            "parameters": {"cell_format": "CR2032"},
            "ports": {"positive": "VBAT", "negative": "GND"},
        },
        {
            "id": "power_conversion",
            "sheet": "A",
            "role": "regulator",
            "family": "direct-battery-rail",
            "functional_blocks": ["POWER_CONVERSION"],
            "ports": {"input": "VBAT", "output": "VCC", "ground": "GND"},
        },
    ]
    return state


@pytest.mark.parametrize(
    ("mpn", "footprint", "defect"),
    [
        (
            "BS-07-A1BJ001",
            "Battery:BatteryHolder_MYOUNG_BS-07-A1BJ001_CR2032",
            "model_authored_protected_identity",
        ),
        # An unreviewed holder is not recognized as any requirement's hardware, so
        # it is refused as the unimplemented regulator rather than as a sibling
        # substitution. Either way it never becomes the converter.
        (
            "Keystone3034",
            "Battery:BatteryHolder_Keystone_3034_1x20mm",
            "missing-requirement-implementation",
        ),
    ],
)
@pytest.mark.parametrize("with_support_passive", [False, True])
def test_regulator_unit_rejects_sibling_coin_cell_holder_substitution(
    mpn, footprint, defect, with_support_passive
):
    state = _battery_power_state()
    unit = StageWorkUnit("bom-s001", "bom", "A", requirement_ids=("power_conversion",))
    holder = {
        **_group("power_conversion", "A", prefix="BT"),
        "value": "CR2032 holder",
        "mpn": mpn,
        "symbol": "Device:Battery_Cell",
        "footprint": footprint,
    }
    groups = [holder]
    if with_support_passive:
        # Even an owned-looking label cannot turn an ancillary resistor into
        # the implementation that makes sibling pruning safe.
        groups.append(_group("power_conversion_support", "A"))

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, {"groups": groups}, state, {})

    assert caught.value.defects[defect] == ["power_conversion"]


def test_coin_cell_holder_unit_accepts_own_feature_despite_sibling_holder():
    state = _battery_power_state()
    state["architecture"]["requirements"].append(
        {
            "id": "backup_battery",
            "sheet": "A",
            "role": "power_input",
            "family": "coin-cell-holder",
            "parameters": {"cell_format": "CR2032"},
            "ports": {"positive": "BACKUP", "negative": "GND"},
        }
    )
    unit = StageWorkUnit("bom-r000", "bom", "A", requirement_ids=("battery",))
    holder = {
        **_group("primary_cell", "A", prefix="BT"),
        "value": "CR2032 holder",
        "mpn": "BS-07-A1BJ001",
        "symbol": "Device:Battery_Cell",
        "footprint": "Battery:BatteryHolder_MYOUNG_BS-07-A1BJ001_CR2032",
    }

    validated = validate_unit_candidate(unit, {"groups": [holder]}, state, {})

    assert [group["id"] for group in validated["groups"]] == ["primary_cell"]


def test_regulator_unit_prunes_sibling_holder_only_with_owned_implementation():
    state = _battery_power_state()
    state["architecture"]["requirements"][1].update(
        {"family": "ldo", "exact_part": "MCP1700T-2502E/TT"}
    )
    unit = StageWorkUnit("bom-s001", "bom", "A", requirement_ids=("power_conversion",))
    regulator = {
        **_group("voltage_conditioner", "A", prefix="U"),
        "value": "MCP1700T-2502E/TT",
        "mpn": "MCP1700T-2502E/TT",
        "symbol": "Regulator_Linear:MCP1700-2502E_SOT23",
        "footprint": "Package_TO_SOT_SMD:SOT-23",
    }
    holder = {
        **_group("extra_holder", "A", prefix="BT"),
        "value": "CR2032 holder",
        "mpn": "BS-07-A1BJ001",
        "symbol": "Device:Battery_Cell",
        "footprint": "Battery:BatteryHolder_MYOUNG_BS-07-A1BJ001_CR2032",
    }
    decoupling = {
        **_group("output_decoupling", "A", prefix="C"),
        "value": "1uF",
        "symbol": "Device:C",
        "footprint": "Capacitor_SMD:C_0603_1608Metric",
    }

    validated = validate_unit_candidate(
        unit, {"groups": [regulator, holder, decoupling]}, state, {}
    )

    assert {group["id"] for group in validated["groups"]} == {
        "voltage_conditioner",
        "output_decoupling",
    }


def test_regulator_unit_rejects_empty_direct_battery_rail():
    state = _battery_power_state()
    unit = StageWorkUnit("bom-s001", "bom", "A", requirement_ids=("power_conversion",))

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, {"groups": []}, state, {})

    assert caught.value.defects["empty-sheet"] == ["A"]


@pytest.mark.parametrize("with_sibling", [False, True])
def test_typed_pd_requirement_rejects_ancillary_only_bom_at_origin(with_sibling):
    from kicraft.design.models import CircuitRequirement

    state = _state()
    state["architecture"]["sheets"] = [
        {"name": "A", "function": "USB-C receptacle with PD trigger negotiating 9/12/20 V"}
    ]
    requirements = [
        CircuitRequirement(
            id="pd_trigger_negotiation",
            sheet="A",
            role="power_input",
            family="usb-pd-trigger",
            parameters={"voltages": "9,12,20"},
            ports={"vbus": "VBUS", "cc1": "CC1", "cc2": "CC2"},
        ).model_dump(mode="json")
    ]
    if with_sibling:
        requirements.append(
            CircuitRequirement(
                id="bulk_filter", sheet="A", role="analog_block", family="custom-filter"
            ).model_dump(mode="json")
        )
    state["architecture"]["requirements"] = requirements
    (unit,) = plan_stage_work_units("bom", state, {})
    payload = {
        "groups": [
            _group("pd_trigger_negotiation", "A"),
            {
                **_group("bulk_filter", "A", prefix="C"),
                "value": "10uF",
                "symbol": "Device:C",
                "footprint": "Capacitor_SMD:C_0603_1608Metric",
            },
        ]
    }

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, payload, state, {})

    assert caught.value.defects["missing-requirement-implementation"] == ["pd_trigger_negotiation"]


def test_typed_pd_requirement_preserves_actual_controller_identity():
    from kicraft.design.models import CircuitRequirement

    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": "PD negotiation"}]
    state["architecture"]["requirements"] = [
        CircuitRequirement(
            id="pd_trigger_negotiation",
            sheet="A",
            role="power_input",
            family="usb-pd-trigger",
            parameters={"voltages": "9,12,20"},
        ).model_dump(mode="json")
    ]
    (unit,) = plan_stage_work_units("bom", state, {})
    payload = {
        "groups": [
            {
                **_group("negotiator", "A", prefix="U"),
                "value": "CH224K",
                "mpn": "CH224K",
                "symbol": "ch224k:CH224K",
                "footprint": "ch224k:ESSOP-10_L4.9-W3.9-P1.0-LS6.0-TL-EP",
            },
            _group("configuration_resistor", "A"),
        ]
    }

    validated = validate_unit_candidate(unit, payload, state, {})

    assert {group["id"] for group in validated["groups"]} == {
        "negotiator",
        "configuration_resistor",
    }


def test_bom_unit_accepts_connector_terminal_synonym():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "output_connector",
            "sheet": "A",
            "role": "connector",
            "family": "output_connector",
        }
    ]
    unit = StageWorkUnit(
        "bom-r000",
        "bom",
        "A",
        requirement_ids=("output_connector",),
        owned_roles=("connector",),
    )

    validated = validate_unit_candidate(
        unit,
        {"groups": [_group("output_terminal", "A", prefix="J")]},
        state,
        {},
    )

    assert validated["groups"][0]["id"] == "output_terminal"


def test_bom_unit_discards_only_unowned_protected_sibling():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "pd_trigger_controller",
            "sheet": "A",
            "role": "bus_interface",
            "family": "pd_trigger_controller",
            "exact_part": "TPS25730",
        }
    ]
    unit = StageWorkUnit(
        "bom-r000",
        "bom",
        "A",
        requirement_ids=("pd_trigger_controller",),
        owned_roles=("bus_interface",),
    )
    payload = {
        "groups": [
            {
                "id": "pd_controller",
                "reference_prefix": "U",
                "quantity": 1,
                "value": "TPS25730",
                "symbol": "Test:TPS25730",
                "footprint": "Test:TPS25730",
                "sheet": "A",
            },
            {
                "id": "usb_c_receptacle",
                "reference_prefix": "J",
                "quantity": 1,
                "value": "USB_C_Receptacle_HRO_TYPE-C-31-M-12",
                "symbol": "Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12",
                "footprint": "Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12",
                "sheet": "A",
            },
        ]
    }

    validated = validate_unit_candidate(unit, payload, state, {})

    assert [group["id"] for group in validated["groups"]] == ["pd_controller"]


def test_bom_unit_prunes_exact_recipe_duplicate_but_rejects_conflict():
    state = _state()
    state["architecture"]["recipe_selections"] = [
        {
            "recipe": "esp32-s3-mini-1-minimal@1",
            "instance": "mcu_core",
            "sheets": {"mcu": "A"},
            "port_bindings": {
                "gnd": "GND",
                "vdd": "+3V3",
                "usb_dm": "USB_D_N",
                "usb_dp": "USB_D_P",
            },
            "requirement_ids": ["mcu_core"],
        }
    ]
    unit = StageWorkUnit(
        "bom-r000",
        "bom",
        "A",
        requirement_ids=("novel_analog",),
    )
    duplicate = {
        "id": "duplicate_bulk_capacitor",
        "reference_prefix": "C",
        "quantity": 1,
        "value": "10uF",
        "symbol": "Device:C",
        "footprint": "Capacitor_SMD:C_0603_1608Metric",
        "sheet": "A",
    }

    owned = _group("novel_filter", "A")
    validated = validate_unit_candidate(
        unit,
        {"groups": [duplicate, owned]},
        state,
        {},
    )
    assert [group["id"] for group in validated["groups"]] == ["novel_filter"]

    conflict = {**duplicate, "footprint": "Capacitor_SMD:C_0805_2012Metric"}
    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, {"groups": [conflict, owned]}, state, {})
    assert caught.value.defects["recipe-duplicate"] == ["duplicate_bulk_capacitor"]


def test_bom_unit_returns_bounded_structured_identity_defect(tmp_path, monkeypatch):
    from kicraft.design import cli_app

    monkeypatch.setattr(
        cli_app,
        "_unresolved_footprints",
        lambda bom, project_root: [
            "U1: footprint 'Test:Missing' is not loadable"
            " -- real options: Package_SO:SOIC-8, Package_DIP:DIP-8"
        ],
    )
    monkeypatch.setattr(cli_app, "_unresolved_symbols", lambda bom: [])
    monkeypatch.setattr(
        cli_app,
        "_symbol_footprint_pin_mismatches",
        lambda bom, project_root: [],
    )
    unit = StageWorkUnit("bom-s000", "bom", "A")
    group = {
        "id": "novel_ic",
        "reference_prefix": "U",
        "quantity": 1,
        "value": "NOVEL",
        "symbol": "Test:Novel",
        "footprint": "Test:Missing",
        "sheet": "A",
    }

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(
            unit,
            {"groups": [group]},
            _state(),
            {"_validation_project_root": str(tmp_path)},
        )

    defect = json.loads(caught.value.defects["unresolved-footprint"][0])
    assert defect == {
        "candidates": ["Package_SO:SOIC-8", "Package_DIP:DIP-8"],
        "detail": "U1: footprint 'Test:Missing' is not loadable",
        "field": "footprint",
        "group_id": "novel_ic",
        "rejected_identifier": "Test:Missing",
    }


def test_bom_unit_routes_sourcing_failure_and_pins_to_owning_group(tmp_path, monkeypatch):
    from kicraft.design import cli_app

    monkeypatch.setattr(cli_app, "_unresolved_footprints", lambda bom, project_root: [])
    monkeypatch.setattr(cli_app, "_unresolved_symbols", lambda bom: [])
    monkeypatch.setattr(
        cli_app,
        "_symbol_footprint_pin_mismatches",
        lambda bom, project_root: [],
    )
    unit = StageWorkUnit("bom-s000", "bom", "A")
    group = _group("sourceable_ic", "A")

    def reject_sourcing(bom, project_root):
        return ["R1: MPN DRY is not orderable; in-stock alternates: C123, C456"], []

    monkeypatch.setattr(cli_app, "_resolve_bom_mpn_sourcing", reject_sourcing)
    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(
            unit,
            {"groups": [group]},
            _state(),
            {"_validation_project_root": str(tmp_path)},
        )
    defect = json.loads(caught.value.defects["unresolved-sourcing"][0])
    assert defect["group_id"] == "sourceable_ic"
    assert defect["field"] == "sourcing"
    assert defect["candidates"] == ["C123", "C456"]

    def pin_sourcing(bom, project_root):
        bom.parts[0].sourcing_note = "LCSC C123"
        return [], []

    monkeypatch.setattr(cli_app, "_resolve_bom_mpn_sourcing", pin_sourcing)
    validated = validate_unit_candidate(
        unit,
        {"groups": [group]},
        _state(),
        {"_validation_project_root": str(tmp_path)},
    )
    assert validated["groups"][0]["sourcing_note"] == "LCSC C123"


def test_bom_unit_omits_absent_metadata_sentinels():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    payload = {
        "groups": [
            {
                **_group("resistor", "A"),
                "value": "1k signal termination",
                "mpn": "N/A",
                "datasheet": "unknown",
                "sourcing_note": "none",
            }
        ]
    }

    validated = validate_unit_candidate(unit, payload, _state(), {})

    assert "mpn" not in validated["groups"][0]
    assert "datasheet" not in validated["groups"][0]
    assert "sourcing_note" not in validated["groups"][0]
    assert validated["groups"][0]["symbol"] == "Device:R"
    assert validated["groups"][0]["footprint"] == "Resistor_SMD:R_0603_1608Metric"


def test_bom_value_descriptions_are_not_treated_as_mpns():
    # Models put a value description in `mpn` ("47uH power inductor"); the §9.26
    # sourcing gate rejects that as a non-orderable MPN and hard-fails the unit,
    # so a description (whitespace / identical to the value) is dropped instead.
    unit = StageWorkUnit("bom-s000", "bom", "A")
    payload = {
        "groups": [
            {
                **_group("inductor", "A"),
                "value": "47uH",
                "mpn": "47uH power inductor",
                "sourcing_note": "keep me",
            }
        ]
    }

    validated = validate_unit_candidate(unit, payload, _state(), {})

    assert "mpn" not in validated["groups"][0]
    assert validated["groups"][0]["sourcing_note"] == "keep me"

    kept = validate_unit_candidate(
        unit,
        {"groups": [{**_group("resistor", "A"), "value": "1k", "mpn": "RC0603FR-071KL"}]},
        _state(),
        {},
    )
    assert kept["groups"][0]["mpn"] == "RC0603FR-071KL"


def test_curated_terminal_resolves_its_explicit_manufacturer_identity(tmp_path):
    unit = StageWorkUnit("bom-s000", "bom", "A")
    payload = {
        "groups": [
            {
                **_group("terminal", "A", prefix="J"),
                "value": "2-pin screw terminal",
                "mpn": "WJ126V-5.0-02P-14-00A",
                "symbol": "Connector_Generic:Conn_01x02",
                "footprint": "TerminalBlock:TerminalBlock_bornier-2_P5.08mm",
            }
        ]
    }

    validated = validate_unit_candidate(
        unit, payload, _state(), {"_validation_project_root": str(tmp_path)}
    )

    group = validated["groups"][0]
    assert group["symbol"].startswith("screw-terminal-5mm-2p:")
    assert group["footprint"].startswith("screw-terminal-5mm-2p:")
    assert group["mpn"] == "WJ126V-5.0-02P-14-00A"


def test_terminal_normalization_preserves_real_contact_count(tmp_path):
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    unit = StageWorkUnit("bom-s000", "bom", "A")
    payload = {
        "groups": [
            {
                **_group("terminal", "A", prefix="J"),
                "value": "ScrewTerminal_1x03",
                "symbol": "Connector:Screw_Terminal_01x03",
                "footprint": (
                    "TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-3_1x03_P5.00mm_Horizontal"
                ),
            }
        ]
    }
    validated = validate_unit_candidate(
        unit, payload, _state(), {"_validation_project_root": str(tmp_path)}
    )
    assert {pin["number"] for pin in lookup_pins(validated["groups"][0]["symbol"])["pins"]} == {
        "1",
        "2",
        "3",
    }


def test_curated_namespace_cannot_replace_an_explicitly_selected_device(monkeypatch):
    from types import SimpleNamespace

    from kicraft.server.stage_contracts import _requirement_owns_protected_group
    from kicraft.server import stage_work_units

    conflicting = SimpleNamespace(
        manifest=SimpleNamespace(
            name="Sensor",
            symbol_name="BME280",
            footprint_name="Bosch_LGA-8_2.5x2.5mm_P0.65mm",
            mpn="K2-1102DP-C4SW-04",
            sourcing={},
        )
    )
    monkeypatch.setattr(
        stage_work_units,
        "_curated_part_indexes",
        lambda: ({"Sensor": conflicting}, {}),
    )
    group = BomComponentGroup.model_validate(
        {
            **_group("sensor", "A", prefix="U"),
            "value": "BME280",
            "mpn": "BME280",
            "symbol": "Sensor:BME280",
            "footprint": "Package_LGA:Bosch_LGA-8_2.5x2.5mm_P0.65mm",
        }
    )
    normalized = stage_work_units._normalize_curated_group_identities([group])[0]
    assert _requirement_owns_protected_group(normalized, [{"exact_part": "BME280"}])


def test_curated_prefix_cannot_change_an_explicit_order_code(monkeypatch):
    from types import SimpleNamespace

    from kicraft.server import stage_work_units

    carrier = SimpleNamespace(
        manifest=SimpleNamespace(
            name="nrf52840",
            symbol_name="NRF52840-QIAA-R",
            footprint_name="aQFN-73",
            mpn="NRF52840-QIAA-R",
            sourcing={"lcsc": "C190794"},
        )
    )
    monkeypatch.setattr(
        stage_work_units,
        "_curated_part_indexes",
        lambda: ({"nrf52840": carrier}, {"nrf52840qiaar": carrier}),
    )
    group = BomComponentGroup.model_validate(
        {
            **_group("radio", "A", prefix="U"),
            "value": "NRF52840-QIAA-R7",
            "mpn": "NRF52840-QIAA-R7",
            "symbol": "nrf52840:NRF52840-QIAA-R",
            "footprint": "nrf52840:aQFN-73",
        }
    )
    normalized = stage_work_units._normalize_curated_group_identities([group])[0]
    assert normalized.mpn == "NRF52840-QIAA-R7"
    assert normalized.sourcing_note == group.sourcing_note


def test_curated_generic_pad_preserves_absent_manufacturer_identity(tmp_path):
    unit = StageWorkUnit("bom-s000", "bom", "A")
    payload = {
        "groups": [
            {
                **_group("pad", "A", prefix="J"),
                "value": "CASTELLATED PAD",
                "symbol": "castellated-pad-2p54:Conn_01x01",
                "footprint": "castellated-pad-2p54:Castellated_Pad_2.54mm",
            }
        ]
    }

    validated = validate_unit_candidate(
        unit, payload, _state(), {"_validation_project_root": str(tmp_path)}
    )

    assert validated["groups"][0]["symbol"] == payload["groups"][0]["symbol"]
    assert "mpn" not in validated["groups"][0]


def test_wiring_unit_deduplicates_only_identical_assignments():
    state = _state([{"ref": "R1", "sheet": "A", "symbol": "Device:R"}])
    unit = StageWorkUnit(
        "wiring-u000",
        "wiring",
        "A",
        refs=("R1",),
        expected_pins=(("R1", "1"),),
    )
    extras = {"symbol_pinouts": {"R1": _pinout(1)}}

    validated = validate_unit_candidate(
        unit,
        {
            "pins": [
                {"ref": "R1", "pin": "1", "net": "SIG"},
                {"ref": "R1", "pin": "1", "net": "SIG"},
            ]
        },
        state,
        extras,
    )

    assert validated["pins"] == [{"ref": "R1", "pin": "1", "net": "SIG"}]
    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(
            unit,
            {
                "pins": [
                    {"ref": "R1", "pin": "1", "net": "SIG"},
                    {"ref": "R1", "pin": "1", "net": "OTHER"},
                ]
            },
            state,
            extras,
        )
    assert caught.value.defects["duplicate"] == ["R1.1"]
    assert caught.value.defects["expected-pin-set"] == ["R1.1"]
    assert caught.value.defects["rejected-assignment-set"] == [
        '{"net":"OTHER","pin":"1","ref":"R1"}'
    ]


def test_wiring_planner_preserves_bom_unit_boundaries():
    parts = [
        {
            "ref": ref,
            "sheet": "A",
            "symbol": "Device:R",
            "resolution_id": resolution_id,
        }
        for ref, resolution_id in (
            ("R1", "bom-s000"),
            ("R2", "bom-s000"),
            ("R3", "bom-s001"),
        )
    ]
    state = _state(parts)
    extras = {"symbol_pinouts": {part["ref"]: _pinout(2) for part in parts}}

    units = plan_stage_work_units("wiring", state, extras)

    assert [unit.refs for unit in units] == [("R1", "R2"), ("R3",)]


def test_wiring_planner_rejects_overlapping_locked_ownership():
    state = _state([{"ref": "R1", "sheet": "A", "symbol": "Device:R"}])
    extras = {
        "symbol_pinouts": {"R1": _pinout(2)},
        "locked_pin_assignments": [{"ref": "R1", "pin": "1", "net": "GND"}],
        "locked_no_connect_pins": [{"ref": "R1", "pin": "1"}],
    }

    with pytest.raises(ValueError, match="overlaps locked"):
        plan_stage_work_units("wiring", state, extras)


def test_deterministic_sheet_missing_signal_fails_during_wiring_planning():
    state = _state(
        [
            {"ref": "U1", "sheet": "A", "symbol": "X:U", "resolution_source": "recipe"},
            {"ref": "J1", "sheet": "B", "symbol": "X:J", "resolution_source": "llm"},
        ]
    )
    state["architecture"]["inter_sheet_nets"] = [
        {"name": "USB_D_P", "endpoints": [{"sheet": "A"}, {"sheet": "B"}]}
    ]
    state["bom"]["recipe_ownership"] = [
        {
            "recipe": "core",
            "refs": ["U1"],
            "pins": [
                {"ref": "U1", "pin": "1", "net": "GND"},
                {"ref": "U1", "pin": "2", "net": None},
            ],
        }
    ]
    extras = {"symbol_pinouts": {"U1": _pinout(2), "J1": _pinout(2)}}

    with pytest.raises(ValueError, match="deterministic_endpoint_unrealizable.*A:USB_D_P"):
        plan_stage_work_units("wiring", state, extras)


def test_bom_only_lowerer_pins_remain_model_owned_for_wiring():
    state = _state(
        [
            {
                "ref": "J1",
                "sheet": "A",
                "symbol": "usb-type-c-16p:TYPE-C-31-M-12",
                "resolution_source": "lowerer",
            },
            {"ref": "J2", "sheet": "B", "symbol": "X:J", "resolution_source": "llm"},
        ]
    )
    state["architecture"]["inter_sheet_nets"] = [
        {"name": "SEL_1", "endpoints": [{"sheet": "A"}, {"sheet": "B"}]}
    ]
    extras = {"symbol_pinouts": {"J1": _pinout(16), "J2": _pinout(3)}}

    units = plan_stage_work_units("wiring", state, extras)

    assert units[0].planned_resolution_source == "llm"
    assert units[0].expected_pins == tuple(("J1", str(pin)) for pin in range(1, 17))
    assert deterministic_wiring_candidate(units[0], state, extras) is None


def test_c3_gpio_allocations_and_native_usb_reach_locked_wiring_once():
    from kicraft.design.models import CircuitRequirement, RecipeSelection
    from kicraft.design.recipes import expand_recipe, get_recipe
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins
    from kicraft.server.stage_contracts import _normalize_stage_response

    definition = get_recipe("esp32-c3-mini-1-minimal@1")
    ports = {
        "vdd": "+3V3",
        "gnd": "GND",
        "usb_dm": "USB_D_N",
        "usb_dp": "USB_D_P",
        "gpio0": "GPIO0",
        "gpio1": "GPIO1",
    }
    requirement = CircuitRequirement(
        id="core", sheet="A", role="mcu_core", family=definition.family, ports=ports
    )
    selection = RecipeSelection(
        recipe=definition.recipe,
        instance="core",
        sheets={"mcu": "A"},
        parameters={"native_usb": True},
        port_bindings={key: value for key, value in ports.items() if not key.startswith("gpio")},
        requirement_ids=["core"],
        pin_allocations=allocate_requirement_pins(definition, requirement),
    )
    expansion = expand_recipe(selection)
    state = _state([part.model_dump(mode="json") for part in expansion.parts])
    state["architecture"]["sheets"] = [{"name": "A"}]
    state["architecture"]["inter_sheet_nets"] = [
        {"name": net, "endpoints": [{"sheet": "A"}]}
        for net in ("GPIO0", "GPIO1", "USB_D_N", "USB_D_P")
    ]
    state["bom"]["recipe_ownership"] = [expansion.ownership.model_dump(mode="json")]
    inventory = {}
    for row in expansion.ownership.pins:
        inventory.setdefault(row.ref, {"pins": []})["pins"].append({"number": row.pin})

    units = plan_stage_work_units("wiring", state, {"symbol_pinouts": inventory})
    assert units == ()
    merged, _ = _normalize_stage_response("wiring", {"pins": []}, state)
    carried = {
        connection["net_name"]: [
            (endpoint["ref"], endpoint["pin"]) for endpoint in connection["endpoints"]
        ]
        for connection in merged["connections"]
    }
    assert {net: carried[net] for net in ("GPIO0", "GPIO1", "USB_D_N", "USB_D_P")} == {
        "GPIO0": [("U1", "12")],
        "GPIO1": [("U1", "13")],
        "USB_D_N": [("U1", "26")],
        "USB_D_P": [("U1", "27")],
    }


@pytest.mark.parametrize("membership", [[], ["USB_C_RECEPTACLE"]])
def test_recipe_covered_merged_sheet_rejects_missing_functional_ownership(membership):
    state = _state()
    state["functional_spec"] = {
        "blocks": [
            {"name": "USB_C_RECEPTACLE", "category": "interface", "purpose": "USB breakout"},
            {"name": "BREAKOUT_HEADER", "category": "interface", "purpose": "Expose USB pins"},
        ]
    }
    state["architecture"].update(
        sheets=[
            {
                "name": "USB BREAKOUT",
                "stem": "USB_BREAKOUT",
                "function": "USB-C receptacle and signal breakout header",
            }
        ],
        requirements=[
            {
                "id": "usb",
                "sheet": "USB BREAKOUT",
                "role": "power_input",
                "family": "usb-c-power-sink",
                "functional_blocks": membership,
            }
        ],
        recipe_selections=[
            {
                "recipe": "usb-c-5v-sink@1",
                "instance": "usb",
                "sheets": {"power": "USB BREAKOUT"},
                "port_bindings": {"vbus": "VBUS", "gnd": "GND"},
                "requirement_ids": ["usb"],
            }
        ],
    )

    with pytest.raises(ValueError, match="BREAKOUT_HEADER"):
        plan_stage_work_units("bom", state, {})


@pytest.mark.parametrize("lower_header", [False, True])
def test_recipe_model_and_lowerer_functions_survive_one_sheet_bom_merge(lower_header):
    state = _state()
    state["functional_spec"] = {
        "blocks": [
            {"name": name, "category": "interface", "purpose": name}
            for name in ("USB", "HEADER", "CONTROL")
        ]
    }
    state["architecture"].update(
        sheets=[{"name": "A", "stem": "A", "function": "USB power and exposed control header"}],
        requirements=[
            {
                "id": "usb",
                "sheet": "A",
                "role": "power_input",
                "family": "usb-c-power-sink",
                "functional_blocks": ["USB"],
            },
            {
                "id": "header",
                "sheet": "A",
                "role": "connector",
                "family": "pin-header",
                "functional_blocks": ["HEADER"],
                "ports": {"pin1": "SELECT", "pin2": "GND"} if lower_header else {},
            },
            {
                "id": "control",
                "sheet": "A",
                "role": "power_input",
                "family": "usb-pd-selectable-trigger",
                "functional_blocks": ["CONTROL"],
            },
        ],
        recipe_selections=[
            {
                "recipe": "usb-c-5v-sink@1",
                "instance": "usb",
                "sheets": {"power": "A"},
                "port_bindings": {"vbus": "VBUS", "gnd": "GND"},
                "requirement_ids": ["usb"],
            }
        ],
        unresolved_requirement_ids=["header", "control"],
    )

    units = plan_stage_work_units("bom", state, {})
    assert [unit.requirement_ids for unit in units] == (
        [("header",), ("control",)] if lower_header else [("header", "control")]
    )
    candidates = {}
    for unit in units:
        groups = []
        if "control" in unit.requirement_ids:
            groups.append(
                {
                    **_group("negotiator", "A", prefix="U"),
                    "value": "CH224K",
                    "symbol": "ch224k:CH224K",
                    "footprint": "ch224k:ESSOP-10_L4.9-W3.9-P1.0-LS6.0-TL-EP",
                }
            )
        if not lower_header:
            groups.append(
                {
                    **_group("header", "A", prefix="J"),
                    "symbol": "Connector_Generic:Conn_01x02",
                    "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
                }
            )
        candidates[unit.unit_id] = validate_unit_candidate(unit, {"groups": groups}, state, {})

    merged, ref_to_unit, _, _ = merge_bom_units(units, candidates, state)

    assert {group["symbol"] for group in merged["groups"]} == {
        "Connector_Generic:Conn_01x02",
        "ch224k:CH224K",
    }
    assert set(ref_to_unit.values()) == {unit.unit_id for unit in units}


def test_typed_header_cannot_be_replaced_by_passive_footprint():
    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": "External interface"}]
    state["architecture"]["requirements"] = [
        {"id": "header", "sheet": "A", "role": "connector", "family": "pin-header"}
    ]
    (unit,) = plan_stage_work_units("bom", state, {})
    group = {
        **_group("header", "A", prefix="J"),
        "symbol": "Connector_Generic:Conn_01x02",
        "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
    }
    group["footprint"] = _group("passive", "A")["footprint"]

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, {"groups": [group]}, state, {})

    assert caught.value.defects["missing-requirement-implementation"] == ["header"]


def test_fpc_header_misimplementation_recovers_real_24_contact_connector():
    from kicraft.design.part_identity import reviewed_part
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": "FPC breakout"}]
    state["architecture"]["requirements"] = [
        {
            "id": "fpc_connector",
            "sheet": "A",
            "role": "connector",
            "family": "fpc-connector",
            "parameters": {"pitch_mm": 0.5},
            "ports": {f"pin{index}": f"NET{index}" for index in range(1, 25)},
        }
    ]
    (unit,) = plan_stage_work_units("bom", state, {})
    header_group = {
        **_group("header", "A", prefix="J"),
        "value": "PinHeader_1x24",
        "symbol": "Connector_Generic:Conn_01x24",
        "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x24_P2.54mm_Vertical",
    }
    validated = validate_unit_candidate(unit, {"groups": [header_group]}, state, {})
    actual = validated["groups"]
    reviewed = reviewed_part("KH-FG0.5-H2.0-24PIN")
    assert [(group["symbol"], group["footprint"], group["quantity"]) for group in actual] == [
        (reviewed.symbol, reviewed.footprint, 1)
    ]
    assert {pin["number"] for pin in lookup_pins(actual[0]["symbol"])["pins"]} == {
        str(pin) for pin in range(1, 25)
    }


@pytest.mark.parametrize("multiple_requirements", [False, True])
def test_model_owned_header_constraints_cannot_fall_back_to_sheet_prose(multiple_requirements):
    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": "8-pin breakout header"}]
    state["architecture"]["requirements"] = [
        {
            "id": "header",
            "sheet": "A",
            "role": "connector",
            "family": "breakout-header",
            "parameters": {"pin_count": 10},
            "ports": {"vbus": "VBUS", "ground": "GND"},
        }
    ]
    if multiple_requirements:
        state["architecture"]["requirements"].append(
            {"id": "socket", "sheet": "A", "role": "connector", "family": "custom-socket"}
        )
    (unit,) = plan_stage_work_units("bom", state, {})

    assert deterministic_bom_candidate(unit, state) is None
    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(unit, {"groups": []}, state, {})
    assert caught.value.defects["empty-sheet"] == ["A"]


@pytest.mark.parametrize(
    "family", ["usb-pd-trigger", "usb-pd-fixed-trigger", "usb-pd-selectable-trigger"]
)
def test_pd_controller_label_and_substitution_cannot_turn_a_header_into_an_ic(family):
    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": "Selectable supply"}]
    state["architecture"]["requirements"] = [
        {"id": "control", "sheet": "A", "role": "power_input", "family": family}
    ]
    (unit,) = plan_stage_work_units("bom", state, {})
    assert deterministic_bom_candidate(unit, state) is None

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(
            unit,
            {
                "groups": [
                    {
                        **_group("control", "A", prefix="U"),
                        "value": "PD controller replacement",
                        "symbol": "Connector_Generic:Conn_01x02",
                        "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
                    }
                ],
                "substitutions": [
                    {
                        "wanted": "PD controller",
                        "got": "2-pin header",
                        "reason": "Controller implementation placeholder",
                    }
                ],
                "_lowering_requirement_id": "control",
            },
            state,
            {},
        )

    assert caught.value.defects["missing-requirement-implementation"] == ["control"]


def test_a_count_two_requirements_share_is_one_demand_not_two():
    """The brief's own count is one demand on the unit, not one per requirement carrying it.

    Live seed-43 run (2026-09-25): a sheet carried two JST-XH connector requirements -- each
    carrying the intent's "two JST-XH connectors" row, which the architecture contract allows --
    and the parts stage refused all six of its repairs. Two single connectors read
    "jst2: requires 2 real jst-xh-connector, found 0" and one group of two read "found -1",
    because the count was charged against each requirement in turn; the stage then failed with
    ``unit_repair_exhausted`` and the run stopped there.
    """
    from kicraft.design.part_identity import reviewed_part

    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": "Two external connectors"}]
    obligations = [
        {"kind": "physical", "original_obligation_id": "jst", "component_class": "jst-xh-connector"},
        {"kind": "quantity", "original_obligation_id": "two_jst", "subject": "jst-xh connector",
         "minimum": 2},
    ]
    state["architecture"]["requirements"] = [
        {"id": "jst1", "sheet": "A", "role": "connector", "family": "jst-xh-connector",
         "obligations": [dict(row) for row in obligations]},
        {"id": "jst2", "sheet": "A", "role": "connector", "family": "jst-xh-connector",
         "obligations": [dict(row) for row in obligations]},
    ]
    unit = StageWorkUnit("bom-s003", "bom", "A", requirement_ids=("jst1", "jst2"))
    part = reviewed_part("B2B-XH-A(LF)(SN)")
    assert part is not None

    def connectors(*counts: int) -> dict:
        return {
            "groups": [
                {
                    **_group(f"connector{index}", "A", prefix="J", quantity=count),
                    "value": "B2B-XH-A(LF)(SN)",
                    "mpn": "B2B-XH-A(LF)(SN)",
                    "symbol": part.symbol,
                    "footprint": part.footprint,
                }
                for index, count in enumerate(counts, start=1)
            ]
        }

    # The two ways a two-connector sheet is written are the same board.
    assert validate_unit_candidate(unit, connectors(2), state, {})["groups"][0]["quantity"] == 2
    assert len(validate_unit_candidate(unit, connectors(1, 1), state, {})["groups"]) == 2

    # One connector for a two-connector demand is still refused, and the count reads as one.
    with pytest.raises(WorkUnitValidationError) as refused:
        validate_unit_candidate(unit, connectors(1), state, {})
    rows = refused.value.defects["physical-obligation-unfulfilled"]
    assert any("requires 2 real jst-xh-connector, found 1" in row for row in rows)
    assert not any("found -" in row for row in rows)


def test_physical_obligation_requires_real_connector_class_and_count():
    from kicraft.design.part_identity import reviewed_part

    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": "Input and output connectors"}]
    state["architecture"]["requirements"] = [
        {
            "id": "panel",
            "sheet": "A",
            "role": "connector",
            "family": "custom-input-panel",
            "obligations": [
                {
                    "kind": "physical",
                    "original_obligation_id": "bnc",
                    "component_class": "bnc-connector",
                },
                {
                    "kind": "quantity",
                    "original_obligation_id": "bnc_count",
                    "subject": "bnc-connector",
                    "minimum": 2,
                },
            ],
        }
    ]
    unit = StageWorkUnit("bom-panel", "bom", "A", requirement_ids=("panel",))
    part = reviewed_part("KH-BNC50-3511")
    group = {
        **_group("coaxial", "A", prefix="J"),
        "quantity": 2,
        "value": "KH-BNC50-3511",
        "mpn": "KH-BNC50-3511",
        "symbol": part.symbol,
        "footprint": part.footprint,
    }
    assert (
        validate_unit_candidate(unit, {"groups": [group]}, state, {})["groups"][0]["quantity"] == 2
    )
    for changed in (
        {**group, "quantity": 1},
        {
            **group,
            "mpn": None,
            "value": "header",
            "symbol": "Connector_Generic:Conn_01x02",
            "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
        },
    ):
        with pytest.raises(WorkUnitValidationError) as rejected:
            validate_unit_candidate(unit, {"groups": [changed]}, state, {})
        assert rejected.value.defects["physical-obligation-unfulfilled"]


def _usb_breakout_wiring_state():
    from kicraft.design.lowering import lower_requirement

    requirement = {
        "id": "usb",
        "sheet": "A",
        "role": "connector",
        "family": "usb-c-breakout",
        "ports": {"vbus": "VBUS", "gnd": "GND"},
    }
    artifact = lower_requirement(requirement)
    state = _state(
        [
            {
                "ref": "J1",
                "sheet": "A",
                "symbol": artifact.groups[0].symbol,
                "resolution_source": "lowerer",
                "resolution_id": artifact.lowerer_id,
                "lowering_requirement_id": "usb",
                "lowering_role": "connector",
            }
        ]
    )
    state["architecture"]["requirements"] = [requirement]
    pins = tuple(("J1", pin.pin) for pin in (*artifact.pins, *artifact.no_connects))
    extras = {"symbol_pinouts": {"J1": {"pins": [{"number": pin} for _, pin in pins]}}}
    return state, extras, pins


@pytest.mark.parametrize("omitted_pin", ["A4B9", "A5"])
def test_deterministic_wiring_rejects_dropped_connections_and_no_connects(omitted_pin):
    state, extras, pins = _usb_breakout_wiring_state()
    expected = tuple(pin for pin in pins if pin != ("J1", omitted_pin))
    unit = StageWorkUnit("wiring-u000", "wiring", "A", refs=("J1",), expected_pins=expected)

    with pytest.raises(WorkUnitValidationError) as caught:
        deterministic_wiring_candidate(unit, state, extras)

    assert caught.value.defects["unexpected"] == [f"J1.{omitted_pin}"]


def test_deterministic_wiring_preserves_excluded_recipe_connections_and_no_connects():
    state, extras, pins = _usb_breakout_wiring_state()
    extras["locked_pin_assignments"] = [{"ref": "J1", "pin": "A4B9", "net": "VBUS"}]
    extras["locked_no_connect_pins"] = [{"ref": "J1", "pin": "A5"}]
    (unit,) = plan_stage_work_units("wiring", state, extras)

    candidate = deterministic_wiring_candidate(unit, state, extras)
    rows = {(row["ref"], row["pin"]): row for row in candidate["pins"]}

    assert set(rows) == set(pins) - {("J1", "A4B9"), ("J1", "A5")}
    assert rows[("J1", "B4A9")]["net"] == "VBUS"
    assert rows[("J1", "B5")]["no_connect"] is True


def test_model_owned_led_unit_cannot_use_zero_ohm_current_limiters():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "led",
            "sheet": "A",
            "role": "driver",
            "family": "led-current-resistor",
            "parameters": {"rail_voltage": 3.3, "led_vf": 2.0, "target_current_ma": 5},
            "ports": {"drive": "LED_DRIVE", "gnd": "GND"},
        }
    ]
    unit = StageWorkUnit("bom-s000", "bom", "A", requirement_ids=("led",))
    payload = {
        "groups": [
            {
                **_group("led", "A", prefix="D"),
                "value": "LED",
                "symbol": "Device:LED",
                "footprint": "LED_SMD:LED_0805_2012Metric",
            },
            {**_group("limiter", "A"), "value": "0R"},
        ]
    }
    # The typed lowerer owns this requirement: a 0R model limiter is discarded
    # and the computed nonzero limiter is adopted instead.
    validated = validate_unit_candidate(unit, payload, state, {})
    assert sorted(group["id"] for group in validated["groups"]) == ["led", "resistor"]
    resistor = next(group for group in validated["groups"] if group["id"] == "resistor")
    assert resistor["value"] == "270R"

    payload["groups"][1]["value"] = "270R"
    validated = validate_unit_candidate(unit, payload, state, {})
    assert (
        next(group for group in validated["groups"] if group["id"] == "limiter")["value"] == "270R"
    )


def test_pipeline_authored_bank_is_not_wiped_as_a_sibling_header():
    """A trusted lowerer bank is its own requirement's implementation.

    The servo bank's stock 1x03 groups carry the same 'header' physical feature
    the derived edge pin-header demands on the same recipe sheet, so the unit
    discarded every group and failed ``empty-sheet`` instead of committing the
    bank the lowerer had already built.
    """
    state = _state()
    state["architecture"]["requirements"] = [
        {"id": "mcu_core", "sheet": "A", "role": "mcu_core", "family": "esp32-s3-module"},
        {
            "id": "bank",
            "sheet": "A",
            "role": "connector",
            "family": "servo-connector-bank",
            "parameters": {"channels": 16},
            "ports": {
                "gnd": "GND",
                "vdd": "+5V_SERVO",
                **{f"signal{index}": f"PWM{index}" for index in range(16)},
            },
        },
        {
            "id": "edge",
            "sheet": "A",
            "role": "connector",
            "family": "pin-header",
            "ports": {"pin1": "I2C_SDA", "pin2": "I2C_SCL"},
        },
    ]
    state["architecture"]["recipe_selections"] = [
        {
            "recipe": "esp32-s3-mini-1-minimal@1",
            "instance": "mcu_core",
            "sheets": {"mcu": "A"},
            "requirement_ids": ["mcu_core"],
            "port_bindings": {
                "vdd": "+3V3",
                "gnd": "GND",
                "usb_dm": "USB_DM",
                "usb_dp": "USB_DP",
            },
        }
    ]
    unit = StageWorkUnit("bom-r001", "bom", "A", requirement_ids=("bank",))
    from kicraft.server.stage_work_units import deterministic_bom_candidate

    payload = deterministic_bom_candidate(unit, state)

    validated = validate_unit_candidate(unit, payload, state, {})

    headers = [
        group for group in validated["groups"] if group["symbol"] == "Connector_Generic:Conn_01x03"
    ]
    assert sum(group["quantity"] for group in headers) == 16


def _group_for(identity: str):
    from kicraft.design import part_identity
    from kicraft.design.part_identity import reviewed_part
    from kicraft.server.stage_contracts import BomComponentGroup

    rec = reviewed_part(identity) or next(
        (
            row
            for row in part_identity._STANDARD_LIBRARY_PARTS
            if row.identity.casefold() == identity.casefold()
        ),
        None,
    )
    return BomComponentGroup(
        id="g",
        sheet="MAIN",
        reference_prefix="J",
        quantity=1,
        value=rec.identity,
        symbol=rec.symbol,
        footprint=rec.footprint,
        mpn=rec.identity,
    )


def test_obligation_class_aliases_match_the_reviewed_feature_vocabulary():
    """Obligation classes the model names must reach the reviewed feature vocabulary.

    Only verified pairs are aliased: `usb-connector` is a USB-A part, so it must NOT
    satisfy a USB-C demand.
    """
    from kicraft.server.stage_work_units import _group_has_physical_feature

    usb_c = _group_for("12401610e4#2a")  # feature: usb-c-receptacle
    usb_a = _group_for("u-a-24ss-w-2")  # feature: usb-connector (USB-A)
    assert _group_has_physical_feature(usb_c, "usb-c-connector") is True
    assert _group_has_physical_feature(usb_a, "usb-c-connector") is False
    assert (
        _group_has_physical_feature(_group_for("kh-fg0.5-h2.0-24pin"), "fpc-ffc-connector") is True
    )
    assert _group_has_physical_feature(_group_for("ams1117-5.0"), "voltage-regulator-ic") is True
    # Classes the 2026-09-17 canary demanded under a second spelling of a reviewed class.
    assert _group_has_physical_feature(_group_for("tl3342f260qg"), "momentary-pushbutton") is True
    assert _group_has_physical_feature(_group_for("tl3342f260qg"), "pushbutton") is True
    assert _group_has_physical_feature(_group_for("ltst-c190kgkt"), "led") is True
    assert _group_has_physical_feature(_group_for("u-a-24ss-w-2"), "usb-a-connector") is True
    assert (
        _group_has_physical_feature(_group_for("wj126v-5.0-02p-14-00a"), "power-screw-terminal")
        is True
    )
    # A reviewed 5.00 mm terminal block IS a power connector (LCSC Terminal Blocks,
    # 250 V / 18 A): demanding the generic `power-connector` class for a DC input must
    # not be satisfiable only by the single JST PH record that carries that role.
    assert (
        _group_has_physical_feature(_group_for("wj126v-5.0-02p-14-00a"), "power-connector")
        is True
    )
    assert (
        _group_has_physical_feature(_group_for("wj126v-5.0-04p-14-00a"), "power-connector")
        is True
    )
    # A barrel jack is the corpus's other DC input shape, and both spellings the
    # model uses must reach the one reviewed, catalogued jack (DC005, C431533).
    # Sharing `power-connector` does not make a terminal block a barrel jack.
    assert _group_has_physical_feature(_group_for("dc005"), "barrel-jack-connector") is True
    assert _group_has_physical_feature(_group_for("dc005"), "barrel-jack") is True
    assert _group_has_physical_feature(_group_for("dc005"), "power-connector") is True
    assert (
        _group_has_physical_feature(_group_for("wj126v-5.0-02p-14-00a"), "barrel-jack-connector")
        is False
    )
    # The role stays a role: a part that carries no power path must not answer it.
    assert _group_has_physical_feature(_group_for("grm188r71c104ka01d"), "power-connector") is False
    assert _group_has_physical_feature(_group_for("tps5430ddar"), "buck-converter-ic") is True
    assert _group_has_physical_feature(_group_for("max31855kasa+"), "thermocouple-input") is True
    assert _group_has_physical_feature(_group_for("aonr21357"), "high-side-load-switch") is True
    assert _group_has_physical_feature(_group_for("pc817c-s"), "opto-isolator") is True
    assert _group_has_physical_feature(_group_for("dl-rfm95-868m"), "lora-radio-module") is True
    assert _group_has_physical_feature(_group_for("ltst-c190kgkt"), "status-led") is True
    assert _group_has_physical_feature(_group_for("e6c0805wway1uda(1.1t m)"), "power-led") is True
    # An addressable LED is not an indicator LED, and a digital isolator is not an optocoupler.
    assert _group_has_physical_feature(_group_for("ws2812b-b/t"), "status-led") is False
    assert _group_has_physical_feature(_group_for("adum1301arwz-rl"), "opto-isolator") is False


def test_unit_refusal_is_written_as_a_loadable_stage_diagnostic():
    """A unit-stage refusal must survive a state round trip.

    The stage stamp falls back to the drive's bare ``diagnostic`` when a stage produced no
    ``diagnostics`` rows, and that row is typed (``code``/``severity``/``message`` required,
    extra keys forbidden). The raw ``{unit_id, defects}`` map therefore made every state
    saved after a BOM/wiring unit exhausted its repair loop unreadable by
    ``ConversationState`` -- so ``stage_driver replay`` could not open exactly the runs it
    exists to iterate on, and neither could any other validated reader.
    """
    from kicraft.design.models import ConversationState, StageDiagnostic
    from kicraft.server.stage_work_units import WorkUnitValidationError, unit_defect_diagnostic

    error = WorkUnitValidationError(
        "bom-r000",
        {
            "physical-obligation-unfulfilled": [
                "input:dc_input: requires 1 real power-connector, found 0"
            ],
            "empty-sheet": [],
        },
    )
    row = unit_defect_diagnostic(error, str(error))
    diag = StageDiagnostic.model_validate(row)
    # The failing check is the row's code, and the offender text stays readable.
    assert diag.code == "physical_obligation_unfulfilled"
    assert diag.severity == "repair_required"
    assert "power-connector" in diag.message
    assert "input:dc_input: requires 1 real power-connector, found 0" in " ".join(diag.evidence)
    assert "unit bom-r000" in diag.evidence
    # ...and a state carrying it still loads.
    state = ConversationState.model_validate(
        {"stage_status": {"bom": {"diagnostics": [row], "ok": False}}}
    )
    reloaded = ConversationState.model_validate(state.model_dump())
    assert reloaded.stage_status["bom"].diagnostics[0].code == "physical_obligation_unfulfilled"


def test_lowerer_physical_witness_is_explicit_and_requirement_bound():
    """A verified R-2R artifact proves its topology, not arbitrary obligations."""
    from kicraft.server.stage_work_units import _requirement_obligation_defects

    requirement = {
        "id": "ladder",
        "obligations": [
            {
                "kind": "physical",
                "original_obligation_id": "ladder",
                "component_class": "resistor-network",
            }
        ],
    }
    groups = [
        BomComponentGroup(
            id="series",
            sheet="DAC",
            reference_prefix="R",
            quantity=7,
            value="10k",
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
        ),
        BomComponentGroup(
            id="branch",
            sheet="DAC",
            reference_prefix="R",
            quantity=9,
            value="20k",
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
        ),
    ]

    assert (
        _requirement_obligation_defects(
            [requirement],
            groups,
            trusted_lowerer_id="r2r-ladder@1",
            trusted_requirement_id="ladder",
        )["physical-obligation-unfulfilled"]
        == []
    )
    assert _requirement_obligation_defects(
        [
            {
                **requirement,
                "obligations": [
                    {**requirement["obligations"][0], "component_class": "binding-post-terminal"}
                ],
            }
        ],
        groups,
        trusted_lowerer_id="r2r-ladder@1",
        trusted_requirement_id="ladder",
    )["physical-obligation-unfulfilled"]
    assert _requirement_obligation_defects(
        [
            {
                **requirement,
                "obligations": [
                    *requirement["obligations"],
                    {"kind": "quantity", "subject": "resistor-network", "minimum": 2},
                ],
            }
        ],
        groups,
        trusted_lowerer_id="r2r-ladder@1",
        trusted_requirement_id="ladder",
    )["physical-obligation-unfulfilled"]


def test_model_private_lowerer_metadata_cannot_satisfy_a_physical_obligation():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "terminal",
            "sheet": "A",
            "role": "connector",
            "family": "unlowered-terminal",
            "obligations": [
                {
                    "kind": "physical",
                    "original_obligation_id": "thermocouple",
                    "component_class": "thermocouple-input",
                }
            ],
        }
    ]
    unit = StageWorkUnit("bom-r000", "bom", "A", requirement_ids=("terminal",))
    group = {
        **_group("terminal", "A", prefix="J"),
        "value": "WJ126V-5.0-2P",
        "mpn": "WJ126V-5.0-02P-14-00A",
        "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P",
        "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
    }

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(
            unit,
            {
                "groups": [group],
                "_lowerer_id": "screw-terminal@1",
                "_lowering_requirement_id": "terminal",
            },
            state,
            {},
        )

    assert caught.value.defects["physical-obligation-unfulfilled"]


def test_binding_post_order_code_is_an_exact_declared_owner():
    from kicraft.server.stage_work_units import _group_matches_requirement_identity

    requirement = {"id": "terminal", "exact_part": "keystone-8734"}
    binding_post = BomComponentGroup(
        id="terminal",
        sheet="INPUT",
        reference_prefix="J",
        quantity=1,
        value="Keystone 8734",
        symbol="keystone-8734-binding-post:Keystone_8734",
        footprint="keystone-8734-binding-post:Keystone_8734",
        mpn="8734",
    )

    assert _group_matches_requirement_identity(binding_post, requirement)
    assert not _group_matches_requirement_identity(
        binding_post.model_copy(
            update={
                "symbol": "Connector_Generic:Conn_01x01",
                "footprint": "Connector_PinSocket_2.54mm:PinSocket_1x01_P2.54mm_Vertical",
            }
        ),
        requirement,
    )


def test_declared_interface_accepts_only_a_compiler_matched_fpc_owner():
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.models import CircuitRequirement
    from kicraft.server.stage_work_units import _requirement_obligation_defects

    fpc = {
        "id": "fpc",
        "sheet": "BREAKOUT",
        "role": "connector",
        "family": "fpc-connector",
        "parameters": {"pitch_mm": 0.5},
        "ports": {f"pin{index}": f"NET{index}" for index in range(1, 25)},
    }
    breakout = {
        "id": "breakout",
        "ports": {f"contact{index}": f"NET{index}" for index in range(1, 25)},
        "obligations": [],
        "declared_interface": {
            "ports": [{"key": "contact1", "pin": "1", "direction": "bidirectional"}]
        },
    }
    artifact = lower_requirement(CircuitRequirement.model_validate(fpc))
    assert artifact is not None
    lowered = artifact.groups[0]
    owner = BomComponentGroup(
        id="fpc",
        sheet="BREAKOUT",
        reference_prefix=lowered.reference_prefix,
        quantity=lowered.quantity,
        value=lowered.value,
        symbol=lowered.symbol,
        footprint=lowered.footprint,
        mpn=lowered.mpn,
    )

    assert (
        _requirement_obligation_defects([breakout], [owner], compiler_requirements=[fpc])[
            "declared-interface-unrealized"
        ]
        == []
    )
    assert _requirement_obligation_defects(
        [breakout], [owner.model_copy(update={"mpn": "WRONG"})], compiler_requirements=[fpc]
    )["declared-interface-unrealized"] == [
        "breakout: declared interface needs one identified hardware owner"
    ]


def test_unreviewed_part_class_is_proven_by_a_real_resolved_part():
    """A class the library has never covered is satisfied by a real part, not refused.

    Decision 2026-09-18: the reviewed library can only answer for the classes it covers, so a
    demand for a new category (`gps-module`) must be met by a real, resolvable part — exact
    MPN, symbol pin inventory, footprint — instead of blocking every novel design. A label
    with no orderable identity is still not evidence, and a covered class still needs its
    reviewed record.
    """
    from kicraft.server.stage_work_units import _group_has_physical_feature

    resolved = BomComponentGroup(
        id="gnss",
        sheet="MAIN",
        reference_prefix="U",
        quantity=1,
        value="NEO-6M",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric",
        mpn="NEO-6M-0-001",
    )
    assert _group_has_physical_feature(resolved, "gps-module") is True
    assert _group_has_physical_feature(resolved, "air-quality-sensor") is True
    # No orderable identity: the parts stage emitted a label, so the demand stands.
    assert (
        _group_has_physical_feature(resolved.model_copy(update={"mpn": None}), "gps-module")
        is False
    )
    # An unreviewed part never answers for a class the reviewed library does cover.
    assert _group_has_physical_feature(resolved, "usb-c-receptacle") is False


def test_pin_less_declared_interface_claim_is_refused_by_name(monkeypatch):
    """A claim without a pin number cannot be verified; say that, not "pin None".

    The canary (2026-09-17, `lora-node`) reported `mcu:vdd: claimed pin None is not in
    stm32l031…`, which reads as a wrong pin rather than a missing one, while the stage skill
    asked for a claim with no pin at all.
    """
    from kicraft.server.stage_work_units import _requirement_obligation_defects

    symbol = _group_for("stm32l031k6t6").symbol
    requirement = {
        "id": "mcu",
        "family": "stm32l0-mcu",
        "exact_part": "STM32L031K6T6",
        "obligations": [],
        "declared_interface": {
            "ports": [
                {"key": "vdd", "direction": "power", "function": "logic supply"},
                {"key": "pa9", "pin": "99", "direction": "output", "function": "uart tx"},
            ]
        },
    }
    group = _group_for("stm32l031k6t6")

    defects = _requirement_obligation_defects([requirement], [group])[
        "declared-interface-unrealized"
    ]

    assert any("vdd" in row and "states no pin number" in row for row in defects), defects
    assert any("pa9" in row and "claimed pin '99'" in row for row in defects), defects
    assert not any("claimed pin None" in row for row in defects)
    assert symbol


def test_declared_interface_claim_may_name_a_pin_by_its_symbol_name(monkeypatch):
    """A symbol names a contact twice; a claim may use either name.

    The 2026-09-19 baseline's BOM deaths were dominated by correct claims read as wrong pins:
    a draft wrote `pin: "VIN"` for the TPS5430DDA or `"VDD"` for the MCP23017-E_SO, the check
    compared that against the symbol's pin *numbers* and refused every one of them. The
    synthesis side already resolves a claimed selector against the pin names
    (`validation._declared_port_pin`); the BOM realization check must too, and refuse only a
    string that is neither a number nor one unique pin name.
    """
    from kicraft.server.stage_work_units import _requirement_obligation_defects

    group = BomComponentGroup(
        id="mcp",
        sheet="GPIO",
        reference_prefix="U",
        quantity=1,
        value="MCP23017T-E/SS",
        symbol="mcp23017t-e-ss:MCP23017T-E_SS",
        footprint="mcp23017t-e-ss:SSOP-28_L10.2-W5.3-P0.65-LS7.8-BL",
        mpn="MCP23017T-E/SS",
    )
    requirement = {
        "id": "mcp23017",
        "family": "mcp23017",
        "exact_part": "MCP23017T-E/SS",
        "obligations": [],
        "declared_interface": {
            "ports": [
                {"key": "vdd", "pin": "VDD", "direction": "power", "function": "logic supply"},
                {"key": "gnd", "pin": "9", "direction": "power", "function": "ground"},
                {"key": "reset", "pin": "RESET", "direction": "input", "function": "not a pin"},
            ]
        },
    }

    defects = _requirement_obligation_defects([requirement], [group])[
        "declared-interface-unrealized"
    ]

    assert not any("vdd" in row for row in defects), defects
    assert not any("gnd" in row for row in defects), defects
    assert any("reset" in row and "claimed pin 'RESET'" in row for row in defects), defects


def test_unfulfilled_obligation_names_the_groups_the_unit_emitted():
    """The defect must say what the unit *did* emit, not only that a class is missing.

    Every 2026-09-17 canary BOM failure reads `requires 1 real usb-c-receptacle, found 0` while
    naming neither the group the unit emitted nor its identity, so the repair (and the next
    diagnosis) cannot tell "the draft picked an unreviewed part" from "the draft emitted no part".
    """
    from kicraft.server.stage_work_units import _requirement_obligation_defects

    requirement = {
        "id": "usb_c",
        "family": "usb-c-breakout",
        "exact_part": "TYPE-C-31-M-12",
        "obligations": [
            {
                "kind": "physical",
                "original_obligation_id": "usb-c-input",
                "component_class": "usb-c-receptacle",
            }
        ],
    }
    unreviewed = _group_for("12401610e4#2a").model_copy(
        update={"id": "usb_c", "symbol": "Connector:USB_C_Receptacle_USB2.0_16P"}
    )

    defects = _requirement_obligation_defects([requirement], [unreviewed])[
        "physical-obligation-unfulfilled"
    ]

    assert len(defects) == 1
    assert "requires 1 real usb-c-receptacle, found 0" in defects[0]
    assert "the unit emitted: usb_c=Connector:USB_C_Receptacle_USB2.0_16P" in defects[0]


def test_board_fact_obligations_are_not_bom_component_demands():
    """A fabrication feature and an absent class never demand a BOM group.

    The demand side reads `physical` rows only (the plan's Phase D step 3). A `negative` row
    naming a class the unit *does* emit must not be reported either: as an unowned demand it
    would read "requires 1 real <class>, found 1", and no group can ever implement an absence.
    """
    from kicraft.server.stage_work_units import _requirement_obligation_defects

    requirement = {
        "id": "led_driver",
        "family": "led-cc-driver",
        "obligations": [
            {
                "kind": "fabrication",
                "original_obligation_id": "copper_heatsink_area",
                "feature": "copper-area",
                "minimum": 300,
                "unit": "mm2",
            },
            {
                "kind": "negative",
                "original_obligation_id": "no_microcontroller",
                "absent_class": "usb-c-receptacle",
            },
        ],
    }
    # The group is a reviewed usb-c-receptacle: exactly the class the negative row forbids.
    present = _group_for("12401610e4#2a")

    defects = _requirement_obligation_defects([requirement], [present])

    assert defects == {
        "physical-obligation-unfulfilled": [],
        "declared-interface-unrealized": [],
    }


def _usb_breakout_state():
    """One family-backed requirement (typed lowerer) with a declared part obligation."""
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "usb",
            "sheet": "A",
            "role": "connector",
            "family": "usb-c-breakout",
            "ports": {"vbus": "VBUS", "gnd": "GND"},
            "obligations": [
                {
                    "kind": "physical",
                    "original_obligation_id": "usb-c-input",
                    "component_class": "usb-c-receptacle",
                }
            ],
        }
    ]
    return state


def _usb_breakout_unit():
    return StageWorkUnit("bom-usb", "bom", "A", requirement_ids=("usb",))


def _unreviewed_usb_receptacle():
    return {
        **_group("usb_c", "A", prefix="J"),
        "value": "USB_C_Receptacle_USB2.0_16P",
        "symbol": "Connector:USB_C_Receptacle_USB2.0_16P",
        "footprint": "Connector_USB:USB_C_Receptacle_USB2.0_16P_P2.50mm",
    }


def _lowered_usb_receptacle():
    from kicraft.design.lowering import lower_requirement

    group = lower_requirement(
        {
            "id": "usb",
            "sheet": "A",
            "role": "connector",
            "family": "usb-c-breakout",
            "ports": {"vbus": "VBUS", "gnd": "GND"},
        }
    ).groups[0]
    return {
        **_group("usb_c", "A", prefix=group.reference_prefix),
        "value": group.value,
        "symbol": group.symbol,
        "footprint": group.footprint,
        "mpn": group.mpn,
    }


def test_real_usb_lowerer_candidate_keeps_an_mpn_equal_to_its_value():
    """A deterministic USB artifact must not lose its true order code in cleanup."""
    unit = _usb_breakout_unit()
    state = _usb_breakout_state()
    state["architecture"]["requirements"][0]["exact_part"] = "12401610E4#2A"
    candidate = deterministic_bom_candidate(unit, state)

    assert candidate is not None
    expected = candidate["groups"][0]
    assert expected["mpn"] == expected["value"] == "12401610E4#2A"

    validated = validate_unit_candidate(unit, candidate, state, {})

    assert validated["groups"][0]["mpn"] == expected["mpn"]


def test_canonical_r2r_realization_requires_the_complete_lowered_topology():
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.models import BOM, Architecture, BomPart, CircuitRequirement, Sheet
    from kicraft.design.synthesis.validation import check_requirement_physical_realization

    requirement = CircuitRequirement.model_validate(
        {
            "id": "r2r",
            "sheet": "DAC",
            "role": "analog_block",
            "family": "r2r-ladder",
            "parameters": {"bits": 2, "r_value": 10000, "two_r_value": 20000},
            "ports": {"bit0": "BIT0", "bit1": "BIT1", "output": "OUT", "gnd": "GND"},
            "obligations": [
                {
                    "kind": "physical",
                    "original_obligation_id": "r2r-network",
                    "component_class": "resistor-network",
                }
            ],
        }
    )
    artifact = lower_requirement(requirement)
    assert artifact is not None
    parts = []
    number = 1
    for group in artifact.groups:
        for index in range(group.quantity):
            parts.append(
                BomPart(
                    ref=f"{group.reference_prefix}{number}",
                    value=group.value,
                    symbol=group.symbol,
                    footprint=group.footprint,
                    sheet="DAC",
                    mpn=group.mpn,
                    resolution_source="lowerer",
                    resolution_id=artifact.lowerer_id,
                    lowering_requirement_id=requirement.id,
                    lowering_role=group.role,
                    lowering_index=index,
                )
            )
            number += 1
    architecture = Architecture(
        sheets=[Sheet(name="DAC", stem="DAC", function="R-2R DAC")],
        power_nets=[],
        inter_sheet_nets=[],
        requirements=[requirement],
    )

    assert check_requirement_physical_realization(architecture, BOM(parts=parts)).ok
    assert not check_requirement_physical_realization(architecture, BOM(parts=parts[:-1])).ok
    assert not check_requirement_physical_realization(
        architecture,
        BOM(parts=[*parts[:-1], parts[-1].model_copy(update={"resolution_source": "llm"})]),
    ).ok


def test_physical_obligation_count_binds_through_the_writers_spelling():
    """Two BNC jacks asked for as "BNC connectors" are still two BNC jacks.

    The count row is prose and the class is kebab-case; 613 of 634 committed intents spelled
    them differently, so the exact-string comparison dropped the count and one jack passed.
    """
    from kicraft.design.part_identity import reviewed_part

    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": "Input and output connectors"}]
    state["architecture"]["requirements"] = [
        {
            "id": "panel",
            "sheet": "A",
            "role": "connector",
            "family": "custom-input-panel",
            "obligations": [
                {
                    "kind": "physical",
                    "original_obligation_id": "bnc",
                    "component_class": "bnc-connector",
                },
                {
                    "kind": "quantity",
                    "original_obligation_id": "bnc_count",
                    "subject": "BNC connectors",
                    "minimum": 2,
                },
            ],
        }
    ]
    unit = StageWorkUnit("bom-panel", "bom", "A", requirement_ids=("panel",))
    part = reviewed_part("KH-BNC50-3511")
    group = {
        **_group("coaxial", "A", prefix="J"),
        "quantity": 2,
        "value": "KH-BNC50-3511",
        "mpn": "KH-BNC50-3511",
        "symbol": part.symbol,
        "footprint": part.footprint,
    }

    assert (
        validate_unit_candidate(unit, {"groups": [group]}, state, {})["groups"][0]["quantity"] == 2
    )
    with pytest.raises(WorkUnitValidationError):
        validate_unit_candidate(unit, {"groups": [{**group, "quantity": 1}]}, state, {})


def test_intent_normalization_reconciles_demanded_classes_and_counts_with_a_decider():
    """The intent normalize step runs one typed decision, and only when a decider is supplied.

    A class the reviewed library cannot read ("jst-xh-connector") and a count whose subject
    names no class ("temperature channels") are the two naming failures that dominated the
    live rounds; both have a closed answer set, so the decider settles them before diagnosis.
    """
    import copy

    from kicraft.server.decision_layer import Answer
    from kicraft.server.reconciliation import shortlist_reviewed_classes
    from kicraft.server.stage_runtime import _normalize_candidate_for_diagnostics

    candidate = {
        "project_stem": "recon",
        "goal": "a small sensor board",
        "constraints": ["3.3 V supply"],
        "assumptions": [],
        "obligations": [
            {
                "kind": "physical",
                "original_obligation_id": "relay",
                "component_class": "through-hole-relay",
            },
            {
                "kind": "quantity",
                "original_obligation_id": "ch",
                "subject": "relay channels",
                "minimum": 4,
            },
            {
                "kind": "physical",
                "original_obligation_id": "host",
                "component_class": "jst-xh-connector",
            },
        ],
        "named_parts": [],
    }
    brief = "a small sensor board"

    baseline = _normalize_candidate_for_diagnostics("intent", copy.deepcopy(candidate), brief, {})
    assert baseline["obligations"][1]["subject"] == "relay channels"
    assert baseline["obligations"][2]["component_class"] == "jst-xh-connector"

    pick = shortlist_reviewed_classes("jst-xh-connector")[0]

    def decider(state, questions):
        return {
            question.key: Answer(
                key=question.key,
                kind=question.kind,
                value=pick if question.key.startswith("class_") else "through-hole-relay",
                confidence=0.9,
            )
            for question in questions
        }

    done = _normalize_candidate_for_diagnostics(
        "intent", copy.deepcopy(candidate), brief, {}, decider=decider
    )
    assert done["obligations"][1]["subject"] == "through-hole-relay"
    assert done["obligations"][2]["component_class"] == pick


def test_a_refused_part_class_gets_a_typed_recommendation_from_the_decider():
    """The repair feedback names the group when exactly one group is the confident reading."""
    from kicraft.server.decision_layer import Answer

    state = _state()
    state["architecture"]["sheets"] = [{"name": "A", "function": "Resistor ladder"}]
    state["architecture"]["requirements"] = [
        {
            "id": "ladder",
            "sheet": "A",
            "role": "analog_block",
            "family": "custom-ladder",
            "obligations": [
                {
                    "kind": "physical",
                    "original_obligation_id": "post",
                    "component_class": "binding-post-terminal",
                }
            ],
        }
    ]
    unit = StageWorkUnit("bom-ladder", "bom", "A", requirement_ids=("ladder",))
    groups = [
        {
            **_group("series", "A", prefix="R"),
            "quantity": 7,
            "value": "10k",
            "symbol": "Device:R",
            "footprint": "Resistor_SMD:R_0603_1608Metric",
        },
        {
            **_group("branch", "A", prefix="R"),
            "quantity": 9,
            "value": "20k",
            "symbol": "Device:R",
            "footprint": "Resistor_SMD:R_0603_1608Metric",
        },
    ]

    # Without a decider the refusal is unchanged: no recommendation is invented.
    with pytest.raises(WorkUnitValidationError) as plain:
        validate_unit_candidate(unit, {"groups": groups}, state, {})
    assert plain.value.defects["physical-obligation-unfulfilled"]
    assert not any("bind the group" in row for row in plain.value.defects["physical-obligation-unfulfilled"])

    def decider(_state, questions):
        return {
            question.key: Answer(
                key=question.key,
                kind=question.kind,
                value=("series" in question.prompt),
                confidence=0.9,
            )
            for question in questions
        }

    with pytest.raises(WorkUnitValidationError) as answered:
        validate_unit_candidate(unit, {"groups": groups}, state, {}, decider=decider)
    rows = answered.value.defects["physical-obligation-unfulfilled"]
    assert any("binding-post-terminal" in row and "series" in row for row in rows), rows


def test_a_claimed_contact_the_symbol_does_not_publish_gets_a_recommendation():
    """A claim naming no published contact gets the reading, so the correction round is precise."""
    from kicraft.server.decision_layer import Answer
    from kicraft.server.stage_work_units import _requirement_obligation_defects

    requirement = {
        "id": "mcu",
        "family": "stm32l0-mcu",
        "exact_part": "STM32L031K6T6",
        "obligations": [],
        "declared_interface": {
            "ports": [{"key": "pa9", "pin": "99", "direction": "output", "function": "uart tx"}]
        },
    }
    group = _group_for("stm32l031k6t6")
    plain = _requirement_obligation_defects([requirement], [group])["declared-interface-unrealized"]
    assert plain and not any("reads as contact" in row for row in plain)

    def decider(_state, questions):
        return {
            question.key: Answer(
                key=question.key,
                kind=question.kind,
                value="1",
                confidence=0.9,
            )
            for question in questions
        }

    rows = _requirement_obligation_defects([requirement], [group], decider=decider)[
        "declared-interface-unrealized"
    ]
    assert any("reads as contact '1'" in row for row in rows), rows


def test_a_group_naming_a_reviewed_part_adopts_that_records_pair():
    """The gate compares the curated pair verbatim, so the pair comes from the record.

    Live walkthrough (2026-09-25): `status_led=Device:LED mpn=LTST-C190KGKT` with a pair the
    record does not publish reported "requires 1 real led, found 0" for seven repair rounds.
    """
    from kicraft.design.part_identity import reviewed_part
    from kicraft.server.stage_work_units import BomComponentGroup, _adopt_reviewed_library_pair

    record = reviewed_part("LTST-C190KGKT")
    group = BomComponentGroup.model_validate(
        {
            "id": "status_led",
            "reference_prefix": "D",
            "quantity": 1,
            "value": "green",
            "sheet": "STATUS INDICATOR",
            "symbol": "Device:LED",
            "footprint": "Device:LED",
            "mpn": "LTST-C190KGKT",
        }
    )
    adopted = _adopt_reviewed_library_pair(group)
    assert (adopted.symbol, adopted.footprint) == (record.symbol, record.footprint)

    # A group whose pair is already the record's, or whose MPN is unknown, is untouched.
    already = group.model_copy(update={"symbol": record.symbol, "footprint": record.footprint})
    assert _adopt_reviewed_library_pair(already) is already
    unknown = group.model_copy(update={"mpn": "NOT-A-REAL-MPN"})
    assert _adopt_reviewed_library_pair(unknown) is unknown


def test_a_connector_requirement_keeps_one_connector_group():
    """Its ports are the connector's contacts, not instances of it.

    Live walkthrough (2026-09-25): each actuator sheet emitted two JST-XH parts, four connectors
    for a brief that names two.
    """
    from kicraft.server.stage_work_units import BomComponentGroup, _keep_one_connector_group

    unit = StageWorkUnit("bom-r001", "bom", "ACTUATOR CONNECTOR 1", requirement_ids=("motor_a",))
    prompt_state = {"architecture": {"requirements": [{"id": "motor_a", "role": "connector"}]}}

    def group(group_id: str) -> BomComponentGroup:
        return BomComponentGroup.model_validate(
            {
                "id": group_id,
                "reference_prefix": "J",
                "quantity": 1,
                "value": "B2B-XH-A(LF)(SN)",
                "symbol": "b2b-xh-a-lf-sn:B2B-XH-A",
                "footprint": "b2b-xh-a-lf-sn:CONN-TH_B2B-XH-A-LF-SN",
                "sheet": "ACTUATOR CONNECTOR 1",
                "mpn": "B2B-XH-A(LF)(SN)",
            }
        )

    kept, dropped = _keep_one_connector_group([group("a"), group("b")], unit, prompt_state)
    assert [row.id for row in kept] == ["a"] and dropped == 1

    # A unit with several identical passives is untouched: only connector groups are deduplicated.
    resistor = BomComponentGroup.model_validate(
        {
            "id": "c1",
            "reference_prefix": "C",
            "quantity": 2,
            "value": "100nF",
            "symbol": "Device:C",
            "footprint": "Capacitor_SMD:C_0603_1608Metric",
            "sheet": "ACTUATOR CONNECTOR 1",
        }
    )
    regulator_unit = StageWorkUnit(
        "bom-s005", "bom", "REG", requirement_ids=("reg",)
    )
    assert _keep_one_connector_group(
        [resistor, resistor.model_copy(update={"id": "c2"})],
        regulator_unit,
        {"architecture": {"requirements": [{"id": "reg", "role": "regulator"}]}},
    )[1] == 0


def test_a_sheet_scoped_connector_unit_keeps_one_connector_too():
    """Units for a sheet with a single connector requirement carry no requirement ids."""
    from kicraft.server.stage_work_units import BomComponentGroup, _keep_one_connector_group

    unit = StageWorkUnit("bom-s002", "bom", "ACTUATOR CONNECTOR 1")
    prompt_state = {
        "architecture": {
            "requirements": [
                {"id": "motor_a", "role": "connector", "sheet": "ACTUATOR CONNECTOR 1"}
            ]
        }
    }
    group = BomComponentGroup.model_validate(
        {
            "id": "motor_a",
            "reference_prefix": "J",
            "quantity": 1,
            "value": "B2B-XH-A(LF)(SN)",
            "symbol": "b2b-xh-a-lf-sn:B2B-XH-A",
            "footprint": "b2b-xh-a-lf-sn:CONN-TH_B2B-XH-A-LF-SN",
            "sheet": "ACTUATOR CONNECTOR 1",
            "mpn": "B2B-XH-A(LF)(SN)",
        }
    )
    kept, dropped = _keep_one_connector_group(
        [group, group.model_copy(update={"id": "motor_a_2"})], unit, prompt_state
    )
    assert [row.id for row in kept] == ["motor_a"] and dropped == 1
