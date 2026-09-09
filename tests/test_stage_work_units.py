from dataclasses import FrozenInstanceError

import pytest

from kicraft.server.stage_work_units import (
    StageDraftStore,
    StageWorkUnit,
    WorkUnitValidationError,
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


def test_bom_unit_rejects_selector_support_without_selector():
    state = _state()
    state["architecture"]["requirements"] = [
        {
            "id": "voltage_selector_switch",
            "sheet": "A",
            "role": "user_io",
            "family": "voltage_selector_switch",
        }
    ]
    unit = StageWorkUnit(
        "bom-r000",
        "bom",
        "A",
        requirement_ids=("voltage_selector_switch",),
        owned_roles=("user_io",),
    )

    with pytest.raises(WorkUnitValidationError) as caught:
        validate_unit_candidate(
            unit,
            {"groups": [_group("jumper", "A", prefix="J")]},
            state,
            {},
        )

    assert caught.value.defects["missing-requirement-implementation"] == ["voltage_selector_switch"]


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
