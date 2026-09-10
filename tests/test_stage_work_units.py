from dataclasses import FrozenInstanceError

import pytest

from kicraft.server.stage_work_units import (
    StageDraftStore,
    StageWorkUnit,
    WorkUnitValidationError,
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


def test_common_standard_sheets_lower_without_provider_guessing():
    state = _state()
    state["architecture"]["sheets"] = [
        {
            "name": "PORT 1",
            "function": "USB-A receptacle with current limiting and status LED",
        },
        {"name": "QSPI FLASH", "function": "QSPI flash storage"},
        {"name": "POWER", "function": "3.3 V LDO regulator"},
        {"name": "GPIO", "function": "Castellated GPIO breakout header"},
    ]
    state["architecture"]["requirements"] = [
        {
            "id": "port",
            "sheet": "PORT 1",
            "role": "driver",
            "family": "current-limit-switch",
        },
        {
            "id": "flash",
            "sheet": "QSPI FLASH",
            "role": "bus_interface",
            "family": "qspi-flash",
        },
        {"id": "ldo", "sheet": "POWER", "role": "regulator", "family": "ldo"},
        {
            "id": "gpio",
            "sheet": "GPIO",
            "role": "connector",
            "family": "gpio-header",
            "parameters": {"pins": 28},
        },
    ]

    expected = {
        "PORT 1": {"usb_a_receptacle", "current_limit_switch", "status_led"},
        "QSPI FLASH": {"qspi_flash"},
        "POWER": {"ldo_3v3", "ldo_caps"},
        "GPIO": {"gpio_header"},
    }
    for index, (sheet, ids) in enumerate(expected.items()):
        requirement_id = state["architecture"]["requirements"][index]["id"]
        candidate = validate_unit_candidate(
            StageWorkUnit(
                f"bom-s{index:03d}",
                "bom",
                sheet,
                requirement_ids=(requirement_id,),
            ),
            {"groups": []},
            state,
            {},
        )
        assert ids <= {group["id"] for group in candidate["groups"]}


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


def test_bom_planning_skips_net_like_power_requirements():
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

    # The plain filter requirement rolls up into a sheet-scoped unit; sheet B
    # (no requirements) still gets its own unit so it can never come back empty.
    assert [(unit.requirement_ids, unit.sheet) for unit in units] == [
        (("filter",), "A"),
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
    merged, refs, lowering = merge_bom_units(units, candidates, state)
    assert [group["id"] for group in merged["groups"]] == ["s000_res", "s001_res"]
    assert merged["arrays"][0]["group_id"] == "s000_res"
    assert merged["assumptions"] == ["shared (defaulted)"]
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


def test_combined_fpc_header_requirements_stay_in_one_sheet_unit():
    state = _state()
    state["architecture"]["sheets"] = [
        {"name": "A", "function": "24-pin FPC breakout to 0.1-inch header"}
    ]
    state["architecture"]["requirements"] = [
        {
            "id": "fpc",
            "sheet": "A",
            "role": "connector",
            "family": "fpc-connector",
            "parameters": {"pitch_mm": 0.5, "pin_count": 24},
            "ports": {"signal": "24x signal"},
        },
        {
            "id": "header",
            "sheet": "A",
            "role": "connector",
            "family": "header",
            "parameters": {"pitch_mm": 2.54, "pin_count": 24},
            "ports": {"signal": "24x signal"},
        },
    ]

    units = plan_stage_work_units("bom", state, {})

    assert len(units) == 1
    assert units[0].requirement_ids == ("fpc", "header")


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


def test_recipe_part_role_omits_redundant_unresolved_requirement():
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

    assert all("xtal" not in unit.requirement_ids for unit in units)


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


def test_bom_unit_allows_owned_protected_identity():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "usb_c_receptacle",
            "sheet": "A",
            "role": "connector",
            "family": "usb_c_receptacle",
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
                "value": "USB_C_Receptacle_HRO_TYPE-C-31-M-12",
                "symbol": "Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12",
                "footprint": "Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12",
                "sheet": "A",
            }
        ]
    }

    validated = validate_unit_candidate(unit, payload, state, {})

    assert validated["groups"][0]["id"] == "usb_c_receptacle"


def test_bom_unit_allows_owned_protected_identity_words():
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
