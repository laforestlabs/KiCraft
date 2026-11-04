"""Consumer-level coverage for the supported intent-to-circuit producer."""
from copy import deepcopy

import pytest

from kicraft.design.architecture_intent import derive_architecture
from kicraft.design.lowering import lower_requirement
from kicraft.design.models import Architecture, BOM
from kicraft.design.recipes.resolver import resolve_architecture_recipes
from kicraft.design.supported_architecture import (
    supported_architecture_intent,
    supported_functional_spec,
)
from kicraft.design.stage_state import invalidate_downstream
from kicraft.design.synthesis.validation import (
    _invalid_lowerer_ownership,
    check_mcu_programming_access,
    check_composed_wiring,
    check_requirement_physical_realization,
)
from kicraft.server.stage_contracts import _normalize_stage_response
from kicraft.server.stage_work_units import probe_architecture_construction


def _physical(identifier, component_class):
    return {"kind": "physical", "original_obligation_id": identifier, "component_class": component_class}


def _intent(family, count, voltage):
    relay = family == "c3"
    item = "relay" if relay else "push-button"
    goal = (f"Build an ESP32-C3 module board with {count} relay outputs and {voltage} V DC input"
            if relay else
            f"Build an ESP32-S3 module temperature board with {count} push buttons and {voltage} V DC input")
    return {
        "goal": goal,
        "constraints": [],
        "named_parts": ["ESP32-C3" if relay else "ESP32-S3"],
        "obligations": [
            _physical("mcu", "microcontroller-module"),
            _physical("item", item),
            _physical("terminal", "screw-terminal"),
            {"kind": "quantity", "original_obligation_id": "count", "subject": item, "minimum": count},
            {"kind": "quantitative", "original_obligation_id": "vin", "quantity": "input voltage", "relation": "equal", "value": voltage, "unit": "V DC"},
        ],
    }


@pytest.mark.parametrize("count", [2, 3, 4])
def test_relay_channels_lower_to_real_isolated_contacts(count):
    intent = _intent("c3", count, 24)
    spec = supported_functional_spec(intent)
    dto = supported_architecture_intent(intent, spec)
    architecture = derive_architecture(dto, spec)
    resolution = resolve_architecture_recipes(architecture, intent)
    assert not resolution.blocking
    req = next(row for row in architecture.requirements if row.family == "relay-output")
    circuit = lower_requirement(req)
    relay = next(group for group in circuit.groups if group.role == "relay")
    terminals = next(group for group in circuit.groups if group.role == "relay_contact_terminal")
    assert relay.quantity == terminals.quantity == count
    assert relay.mpn == "SRD-24VDC-SL-C"
    assert len({req.ports[f"control_{i}"] for i in range(1, count + 1)}) == count
    assert len({req.ports[f"relay_{i}_{contact}"] for i in range(1, count + 1) for contact in ("common", "normally_open", "normally_closed")}) == 3 * count


@pytest.mark.parametrize("count,voltage", [(1, 6), (3, 18), (4, 29)])
def test_temperature_buttons_and_sensor_are_constructive(count, voltage):
    intent = _intent("s3", count, voltage)
    spec = supported_functional_spec(intent)
    dto = supported_architecture_intent(intent, spec)
    architecture = derive_architecture(dto, spec)
    buttons = [row for row in architecture.requirements if row.family == "switch-input"]
    assert len(buttons) == count
    assert len({row.ports["signal"] for row in buttons}) == count
    assert all(lower_requirement(row) is not None for row in buttons)
    sensor = next(row for row in architecture.requirements if row.family == "tmp102-temperature")
    circuit = lower_requirement(sensor)
    assert {group.role for group in circuit.groups} >= {"sensor", "bypass", "sda_pullup", "scl_pullup"}
    assert architecture.rail_voltages["VIN"] == voltage


@pytest.mark.parametrize("extra", ["USB data", "WiFi", "8 MB flash", "analog temperature sensor", "2 A output", "battery input"])
def test_unhandled_user_constraints_decline_instead_of_being_dropped(extra):
    intent = _intent("s3", 3, 18)
    intent["constraints"].append(extra)
    assert supported_functional_spec(intent) is None


@pytest.mark.parametrize("voltage", [5, 30])
def test_converter_input_boundaries_are_conservative(voltage):
    assert supported_functional_spec(_intent("s3", 3, voltage)) is None


def test_existing_incompatible_spec_is_not_rewritten():
    intent = _intent("s3", 3, 18)
    spec = supported_functional_spec(intent)
    incompatible = deepcopy(spec)
    next(block for block in incompatible["blocks"] if block["name"] == "TEMPERATURE")["purpose"] = "Analog temperature sensor output"
    before = deepcopy(incompatible)
    assert supported_architecture_intent(intent, incompatible) is None
    assert incompatible == before


def test_unrecognized_exact_part_is_not_replaced():
    intent = _intent("c3", 2, 24)
    intent["named_parts"].append("ESP32-C3-MINI-1-N4")
    assert supported_functional_spec(intent) is None


def test_bom_acceptance_retains_compiled_graph_but_clears_stale_authored_wiring():
    intent = _intent("c3", 2, 24)
    spec = supported_functional_spec(intent)
    candidate = supported_architecture_intent(intent, spec)
    architecture, _ = _normalize_stage_response(
        "architecture", candidate, {"intent": intent, "functional_spec": spec},
    )
    compiled = probe_architecture_construction({
        "intent": intent, "functional_spec": spec, "architecture": architecture,
    })
    bom = BOM.model_validate(compiled)
    driver = next(part for part in bom.parts if part.mpn == "ULN2003ADR")
    state = {"architecture": architecture, "bom": bom.model_dump()}
    state["bom"]["connections"].append({
        "net_name": "STALE_AUTHORED",
        "sheet": driver.sheet,
        "endpoints": [{"ref": driver.ref, "pin": "1"}],
    })
    invalidate_downstream(state, "bom")
    accepted = BOM.model_validate(state["bom"])
    scope = Architecture.model_validate(architecture)
    assert all(row.net_name != "STALE_AUTHORED" for row in accepted.connections)
    assert _invalid_lowerer_ownership(scope, accepted) == []
    assert check_mcu_programming_access(accepted, architecture=scope).ok

    broken = accepted.model_copy(deep=True)
    for connection in broken.connections:
        connection.endpoints = [
            endpoint for endpoint in connection.endpoints
            if (endpoint.ref, endpoint.pin) != (driver.ref, "16")
        ]
    assert _invalid_lowerer_ownership(scope, broken)


def test_user_button_instances_share_count_without_reusing_recovery_controls():
    intent = _intent("s3", 3, 18)
    spec = supported_functional_spec(intent)
    candidate = supported_architecture_intent(intent, spec)
    architecture, _ = _normalize_stage_response(
        "architecture", candidate, {"intent": intent, "functional_spec": spec},
    )
    bom = BOM.model_validate(probe_architecture_construction({
        "intent": intent, "functional_spec": spec, "architecture": architecture,
    }))
    scope = Architecture.model_validate(architecture)
    assert check_requirement_physical_realization(scope, bom).ok
    user_switches = [
        part for part in bom.parts
        if part.mpn == "KMR232NGULCLFS"
        and (part.lowering_requirement_id or "").startswith("button_")
    ]
    assert len(user_switches) == 3
    missing = bom.model_copy(deep=True)
    missing.parts = [part for part in missing.parts if part.ref != user_switches[0].ref]
    assert not check_requirement_physical_realization(scope, missing).ok


@pytest.mark.parametrize("family,count,voltage", [("c3", 2, 24), ("s3", 3, 18)])
def test_typed_module_rail_and_recovery_requirements_form_a_complete_circuit(family, count, voltage):
    intent = _intent(family, count, voltage)
    intent["obligations"][0]["component_class"] = f"esp32-{family}-module"
    intent["obligations"][-1]["quantity"] = "DC input voltage"
    if family == "c3":
        intent["goal"] += ". Use a reset button"
        intent["obligations"].append(_physical("reset", "tactile-switch"))
    else:
        intent["constraints"].append("Use a four-layer stack-up")
        intent["obligations"].append({
            "kind": "quantitative", "original_obligation_id": "layers",
            "quantity": "PCB copper layers", "relation": "equal",
            "value": 4, "unit": "layers",
        })
        intent["obligations"].append({
            "kind": "quantitative", "original_obligation_id": "bare_layers",
            "quantity": "copper layers", "relation": "equal",
            "value": 4, "unit": "layers",
        })
    spec = supported_functional_spec(intent)
    candidate = supported_architecture_intent(intent, spec)
    architecture, _ = _normalize_stage_response(
        "architecture", candidate, {"intent": intent, "functional_spec": spec},
    )
    bom = BOM.model_validate(probe_architecture_construction({
        "intent": intent, "functional_spec": spec, "architecture": architecture,
    }))
    scope = Architecture.model_validate(architecture)
    assert all(check.ok for check in check_composed_wiring(scope, bom))
    assert {row.original_obligation_id for row in scope.obligations} == {
        row["original_obligation_id"] for row in intent["obligations"]
    }
    missing_core = bom.model_copy(deep=True)
    missing_core.parts = [part for part in missing_core.parts if not (part.mpn or "").startswith("ESP32-")]
    assert not check_requirement_physical_realization(scope, missing_core).ok


@pytest.mark.parametrize("outputs", [2, 4])
def test_extra_output_count_is_not_silently_assumed_to_mean_existing_relays(outputs):
    intent = _intent("c3", 2, 24)
    intent["goal"] += f" and {outputs} outputs"
    assert supported_functional_spec(intent) is None


def test_typed_relay_terminals_are_owned_by_actual_circuit():
    intent = _intent("c3", 2, 24)
    intent["goal"] = ("Design an ESP32-C3 module relay board with two relay outputs, "
                      "a 24 V DC input, and a power LED. Include a reset button.")
    intent["constraints"] = [
        "Relay outputs on screw terminals", "24 V DC input on a screw terminal",
        "Include a power LED", "Include a reset button",
    ]
    intent["obligations"] += [
        _physical("another_terminal", "screw-terminal"),
        {"kind": "quantity", "original_obligation_id": "terminals",
         "subject": "screw-terminal", "minimum": 2},
        _physical("led", "led"), _physical("reset", "pushbutton"),
    ]
    spec = supported_functional_spec(intent)
    candidate = supported_architecture_intent(intent, spec)
    architecture, _ = _normalize_stage_response(
        "architecture", candidate, {"intent": intent, "functional_spec": spec},
    )
    bom = BOM.model_validate(probe_architecture_construction({
        "intent": intent, "functional_spec": spec, "architecture": architecture,
    }))
    scope = Architecture.model_validate(architecture)
    assert _invalid_lowerer_ownership(scope, bom) == []
    assert check_requirement_physical_realization(scope, bom).ok
    assert all(check.ok for check in check_composed_wiring(scope, bom))
    assert len([p for p in bom.parts if p.mpn == "WJ126V-5.0-03P-14-00A"]) == 2
    assert len([p for p in bom.parts if p.mpn == "WJ126V-5.0-02P-14-00A"]) == 1
    assert len([p for p in bom.parts if p.mpn == "SRD-24VDC-SL-C"]) == 2
    intent["obligations"][-3]["minimum"] = 3
    assert supported_functional_spec(intent) is None


def test_board_wide_maximum_preserves_original_fact_and_rejects_conflicts():
    intent = _intent("s3", 3, 18)
    intent["goal"] = ("Build a temperature acquisition board using an ESP32-S3 module, "
                      "an 18 V DC input, and three push buttons.")
    intent["constraints"].append("Four-layer stack-up")
    intent["obligations"].append({
        "kind": "quantitative", "original_obligation_id": "stackup",
        "quantity": "PCB stack-up layers", "relation": "equal", "value": 4, "unit": "layers",
    })
    intent["constraints"].append("Board dimensions must be under 100 x 100 mm")
    intent["obligations"].append({
        "kind": "quantitative", "original_obligation_id": "extent",
        "quantity": "board dimensions", "relation": "maximum", "value": 100, "unit": "mm",
    })
    from kicraft.design.models import board_copper_layers

    spec = supported_functional_spec(intent)
    architecture, _ = _normalize_stage_response(
        "architecture", supported_architecture_intent(intent, spec),
        {"intent": intent, "functional_spec": spec},
    )
    bom = BOM.model_validate(probe_architecture_construction({
        "intent": intent, "functional_spec": spec, "architecture": architecture,
    }))
    scope = Architecture.model_validate(architecture)
    assert all(check.ok for check in check_composed_wiring(scope, bom))
    assert board_copper_layers(scope.obligations) == 4
    intent["obligations"][-1]["value"] = 120
    assert supported_functional_spec(intent) is None
