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


def test_bom_validation_materializes_an_explicit_generic_pin_header():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    state = _state()
    state["architecture"]["topologies"] = {"A": '1x9 0.1" male header'}

    candidate = validate_unit_candidate(unit, {"groups": [], "arrays": []}, state, {})

    assert candidate["groups"] == [
        {
            "id": "generic_pin_header",
            "reference_prefix": "J",
            "quantity": 1,
            "value": "PinHeader_1x09",
            "symbol": "Connector_Generic:Conn_01x09",
            "footprint": (
                "Connector_PinHeader_2.54mm:"
                "PinHeader_1x09_P2.54mm_Vertical"
            ),
            "sheet": "A",
        }
    ]


def test_bom_validation_derives_header_size_from_interface_function():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    state = _state()
    state["architecture"]["topologies"] = {"A": "Passive header interface"}
    state["architecture"]["sheets"][0]["function"] = (
        "Provides eight parallel logic-level inputs with a common ground reference."
    )

    candidate = validate_unit_candidate(unit, {"groups": [], "arrays": []}, state, {})

    assert candidate["groups"][0]["value"] == "PinHeader_1x09"


def test_bom_validation_parses_hyphenated_header_size():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    state = _state()
    state["architecture"]["topologies"] = {"A": "Pin header"}
    state["architecture"]["sheets"][0]["function"] = (
        "Eight-pin header for parallel digital input bits."
    )

    candidate = validate_unit_candidate(unit, {"groups": [], "arrays": []}, state, {})

    assert candidate["groups"][0]["value"] == "PinHeader_1x08"


def test_bom_validation_materializes_low_voltage_opamp_buffer():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    state = _state()
    state["architecture"]["topologies"] = {"A": "Single-supply op-amp voltage follower"}
    state["architecture"]["rail_voltages"] = {"+3V3": 3.3}
    state["architecture"]["sheets"][0]["function"] = "Low-impedance opamp buffer"

    candidate = validate_unit_candidate(unit, {"groups": [], "arrays": []}, state, {})

    assert [group["symbol"] for group in candidate["groups"]] == [
        "mcp6001:MCP6001T-I_OT",
        "Device:C",
    ]



def test_bom_validation_materializes_dimensioned_r2r_ladder():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    state = _state()
    state["architecture"]["topologies"] = {"A": "8-bit R-2R ladder (10k/20k)"}

    candidate = validate_unit_candidate(unit, {"groups": [], "arrays": []}, state, {})

    assert [
        (group["quantity"], group["value"], group["symbol"])
        for group in candidate["groups"]
    ] == [
        (7, "10k", "Device:R"),
        (9, "20k", "Device:R"),
    ]


def test_bom_validation_defaults_r2r_dimensions_from_function():
    unit = StageWorkUnit("bom-s000", "bom", "A")
    state = _state()
    state["architecture"]["topologies"] = {}
    state["architecture"]["sheets"][0]["function"] = (
        "R-2R ladder converts eight digital inputs to analog"
    )
    state["architecture"]["assumptions"] = []
    state["architecture"]["inter_sheet_nets"] = [
        {
            "name": f"D{index}",
            "endpoints": [{"sheet": "A", "direction": "input"}],
        }
        for index in range(8)
    ]


    candidate = validate_unit_candidate(unit, {"groups": [], "arrays": []}, state, {})

    assert [(group["quantity"], group["value"]) for group in candidate["groups"]] == [
        (7, "10k"),
        (9, "20k"),
    ]
    assert candidate["assumptions"] == [
        "R-2R ladder uses 10k/20k resistors (defaulted)"
    ]


def test_bom_validation_derives_interface_pin_count_from_owned_nets():
    state = _state()
    state["architecture"]["topologies"] = {
        "A": "pin header",
        "OUT": "pin header",
    }
    state["architecture"]["sheets"] = [
        {"name": "A", "function": "Digital input header with GND reference"},
        {"name": "OUT", "function": "Single analog output pin"},
    ]
    state["architecture"]["power_nets"] = ["VCC", "GND"]
    state["architecture"]["assumptions"] = [
        "Power input is a 5V header pin on the input header (defaulted)"
    ]
    state["architecture"]["inter_sheet_nets"] = [
        {
            "name": f"D{index}",
            "endpoints": [{"sheet": "A", "direction": "output"}],
        }
        for index in range(8)
    ] + [
        {
            "name": "VOUT",
            "endpoints": [{"sheet": "OUT", "direction": "input"}],
        }
    ]

    input_candidate = validate_unit_candidate(
        StageWorkUnit("bom-s000", "bom", "A"),
        {"groups": [], "arrays": []},
        state,
        {},
    )
    output_candidate = validate_unit_candidate(
        StageWorkUnit("bom-s001", "bom", "OUT"),
        {"groups": [], "arrays": []},
        state,
        {},
    )

    assert input_candidate["groups"][0]["symbol"] == "Connector_Generic:Conn_01x10"
    assert output_candidate["groups"][0]["symbol"] == "Connector_Generic:Conn_01x01"


def test_bom_validation_materializes_selected_mcp1700_power_input():
    state = _state()
    state["architecture"]["topologies"] = {
        "POWER_INPUT": "5V input with 3.3V LDO",
    }
    state["architecture"]["sheets"] = [
        {
            "name": "POWER INPUT",
            "function": "External 5V input with an onboard 3.3V regulator",
        }
    ]
    state["architecture"]["assumptions"] = [
        "Use MCP1700-3.3 and a 2-pin power input header (defaulted)"
    ]

    candidate = validate_unit_candidate(
        StageWorkUnit("bom-s000", "bom", "POWER INPUT"),
        {"groups": [], "arrays": []},
        state,
        {},
    )

    assert [group["symbol"] for group in candidate["groups"]] == [
        "Connector_Generic:Conn_01x02",
        "Regulator_Linear:MCP1700x-330xxTT",
        "Device:C",
    ]


def test_deterministic_opamp_buffer_wiring_uses_real_symbol_pins():
    state = _state(
        [
            {"ref": "U1", "sheet": "A", "symbol": "mcp6001:MCP6001T-I_OT"},
            {"ref": "C1", "sheet": "A", "symbol": "Device:C"},
        ]
    )
    state["architecture"]["power_nets"] = ["5V", "GND"]
    state["architecture"]["inter_sheet_nets"] = [
        {
            "name": "DAC_OUT",
            "endpoints": [{"sheet": "A", "direction": "input"}],
        },
        {
            "name": "ANALOG_OUT",
            "endpoints": [{"sheet": "A", "direction": "output"}],
        },
    ]
    unit = StageWorkUnit(
        "wiring-u000",
        "wiring",
        "A",
        refs=("U1", "C1"),
        expected_pins=tuple(
            [("U1", str(pin)) for pin in range(1, 6)]
            + [("C1", "1"), ("C1", "2")]
        ),
    )
    extras = {
        "symbol_pinouts": {
            "U1": _pinout(5),
            "C1": _pinout(2),
        }
    }

    candidate = deterministic_wiring_candidate(unit, state, extras)

    assert candidate is not None
    by_pin = {(row["ref"], row["pin"]): row["net"] for row in candidate["pins"]}
    assert by_pin == {
        ("U1", "1"): "ANALOG_OUT",
        ("U1", "2"): "GND",
        ("U1", "3"): "DAC_OUT",
        ("U1", "4"): "ANALOG_OUT",
        ("U1", "5"): "5V",
        ("C1", "1"): "5V",
        ("C1", "2"): "GND",
    }


def test_deterministic_r2r_wiring_builds_connected_ladder():
    parts = [
        *[
            {"ref": f"R{index}", "sheet": "A", "symbol": "Device:R", "value": "10k"}
            for index in range(1, 3)
        ],
        *[
            {"ref": f"R{index}", "sheet": "A", "symbol": "Device:R", "value": "20k"}
            for index in range(3, 7)
        ],
    ]
    state = _state(parts)
    state["architecture"]["inter_sheet_nets"] = [
        *[
            {
                "name": f"D{index}",
                "endpoints": [{"sheet": "A", "direction": "input"}],
            }
            for index in range(3)
        ],
        {
            "name": "DAC_OUT",
            "endpoints": [{"sheet": "A", "direction": "output"}],
        },
    ]
    expected = tuple(
        (f"R{ref}", str(pin))
        for ref in range(1, 7)
        for pin in range(1, 3)
    )
    unit = StageWorkUnit(
        "wiring-u000",
        "wiring",
        "A",
        refs=tuple(f"R{ref}" for ref in range(1, 7)),
        expected_pins=expected,
    )
    extras = {
        "symbol_pinouts": {f"R{ref}": _pinout(2) for ref in range(1, 7)}
    }

    candidate = deterministic_wiring_candidate(unit, state, extras)

    assert candidate is not None
    by_pin = {(row["ref"], row["pin"]): row["net"] for row in candidate["pins"]}
    assert by_pin[("R1", "1")] == "DAC_OUT"
    assert by_pin[("R1", "2")] == "R2R_N1"
    assert by_pin[("R3", "1")] == "D2"
    assert by_pin[("R3", "2")] == "DAC_OUT"
    assert by_pin[("R6", "1")] == "R2R_N2"
    assert by_pin[("R6", "2")] == "GND"




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
    merged, refs = merge_bom_units(units, candidates, state)
    assert [group["id"] for group in merged["groups"]] == ["s000_res", "s001_res"]
    assert merged["arrays"][0]["group_id"] == "s000_res"
    assert merged["assumptions"] == ["shared (defaulted)"]
    assert refs == {"R1": "bom-s000", "R2": "bom-s001"}


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
    assert plan_stage_work_units("bom", state, {}) == ()


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
    assert len(units) == 1
    assert units[0].requirement_ids == ("novel_analog",)
    assert units[0].owned_roles == ("analog_block",)
    assert units[0].recipe_ids == ("esp32-s3-mini-1-minimal@1",)


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
    assert caught.value.defects["model_authored_protected_identity"] == [
        "esp32_s3_support"
    ]
