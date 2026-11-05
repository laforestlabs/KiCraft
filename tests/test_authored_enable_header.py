"""An authored EN header must become a complete, unambiguous control circuit."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from kicraft.design.architecture_intent import _adopt_authored_enable_headers


def _intent():
    return {
        "power": {"rails": {"VIN": {"voltage": 5.0, "from": "host.pin1"}}},
        "obligations": [
            {"kind": "physical", "original_obligation_id": "en", "component_class": "jumper"},
            {"kind": "adjustability", "original_obligation_id": "en-control",
             "parameter": "enable", "mechanism": "jumper"},
        ],
        "requirements": [
            {"id": "host", "family": "pin-header", "role": "connector"},
            {"id": "buck", "family": "tlv62569-3v3", "supply": "VIN"},
            {"id": "en", "family": "jumper-header", "exact_part": "TSW-102-07-G-S",
             "sheet": "POWER", "role": "user_io", "obligation_ids": ["en", "en-control"],
             "declared_ports": [{"key": "pin1", "pin": "1"}, {"key": "pin2", "pin": "2"}]},
        ],
        "signals": [{"name": "EN", "from": "en.pin1", "to": "buck.enable"}],
    }


def test_dedicated_authored_header_acquires_real_bias_and_preserves_identity():
    payload = _intent()
    _adopt_authored_enable_headers(payload)
    jumper = payload["requirements"][2]
    assert jumper["family"] == "switch-input"
    assert jumper["exact_part"] == "TSW-102-07-G-S"
    assert jumper["parameters"] == {"pull_policy": "external", "active_level": "high", "voltage": 5.0}
    assert jumper["supply"] == "VIN"
    assert jumper["obligation_ids"] == ["en", "en-control"]
    assert payload["signals"] == [{"name": "EN", "from": "en.signal", "to": "buck.enable"}]
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.models import CircuitRequirement

    control = CircuitRequirement(
        id="en", sheet="POWER", role="user_io", family=jumper["family"],
        exact_part=jumper["exact_part"], parameters=jumper["parameters"],
        ports={"signal": "EN", "vdd": "VIN", "gnd": "GND"},
        obligations=payload["obligations"],
    )
    artifact = lower_requirement(control)
    assert artifact is not None
    assert {(pin.role, pin.pin, pin.net) for pin in artifact.pins} == {
        ("jumper", "1", "EN"), ("jumper", "2", "VIN"),
        ("pulldown", "1", "GND"), ("pulldown", "2", "EN"),
    }


@pytest.mark.parametrize("change", [
    lambda p: p["power"]["rails"]["VIN"].update(voltage=6.0),
    lambda p: p["power"]["rails"]["VIN"].update(voltage=2.4),
    lambda p: p["power"]["rails"]["VIN"].update(**{"from": "buck.output"}),
    lambda p: p["power"]["rails"]["VIN"].update(**{"from": None}),
    lambda p: p["requirements"][1].update(family="tps54331-adjustable"),
    lambda p: p["requirements"][1].update(exact_part="OTHER"),
    lambda p: p["requirements"][2].update(exact_part="OTHER"),
    lambda p: p["requirements"][2].update(ties={"pin2": "GND"}),
    lambda p: p["requirements"][2]["declared_ports"][1].update(supply_rail="VIN"),
    lambda p: p["signals"].append({"name": "OTHER", "from": "en.pin2", "to": "host.pin2"}),
    lambda p: p["signals"][0].update(to=["buck.enable", "host.pin2"]),
    lambda p: p["obligations"][1].update(parameter="output voltage"),
])
def test_ambiguous_or_unsafe_authored_headers_are_not_rewritten(change):
    payload = _intent()
    change(payload)
    before = deepcopy(payload)
    _adopt_authored_enable_headers(payload)
    assert payload == before


def _retained_1148():
    path = Path(__file__).parent / "fixtures/stage_reliability/enable_header_1148_retained.json"
    fixture = json.loads(path.read_text())
    draft = fixture["architecture_draft"]
    regulator = next(row for row in draft["requirements"] if row["id"] == "reg")
    # The provider chose a TPS composition reviewed only above 7 V. This is an
    # explicit internal-controller requalification, not a replay of that choice.
    regulator.update(
        family="tlv62569-3v3", exact_part="TLV62569DBVR",
        parameters={"output_voltage": 3.3},
    )
    draft["assumptions"] = [
        row for row in draft["assumptions"] if "TPS54331" not in row
    ]
    return fixture, draft


def test_retained_null_bearing_enable_completion_is_idempotent():
    from kicraft.design.architecture_intent import complete_architecture_payload

    fixture, draft = _retained_1148()
    draft["obligations"] = fixture["functional_spec"]["obligations"]
    first = complete_architecture_payload(draft)
    snapshot = deepcopy(first)
    second = complete_architecture_payload(first)
    assert second == snapshot
    enable = next(row for row in second["requirements"] if row["id"] == "enable")
    assert enable["family"] == "switch-input"
    assert enable["exact_part"] == "TSW-102-07-G-S"
    assert enable["parameters"] == {"pull_policy": "external", "active_level": "high", "voltage": 5.0}


def test_retained_draft_through_production_normalizer_preserves_connections():
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.models import CircuitRequirement
    from kicraft.server.stage_contracts import _normalize_stage_response

    fixture, draft = _retained_1148()
    architecture, _ = _normalize_stage_response(
        "architecture", draft,
        {"intent": fixture["intent"], "functional_spec": fixture["functional_spec"]},
    )
    requirements = {row["id"]: row for row in architecture["requirements"]}
    control = requirements["enable"]
    assert control["family"] == "switch-input"
    assert control["exact_part"] == "TSW-102-07-G-S"
    assert control["ports"]["signal"] == requirements["reg"]["ports"]["enable"]
    assert control["ports"]["vdd"] == requirements["reg"]["ports"]["input"]
    assert {row["original_obligation_id"] for row in requirements["host"]["obligations"]} == {"host-header"}
    assert "host-header" not in {row["original_obligation_id"] for row in control["obligations"]}
    assert requirements["host"]["ports"]["pin3"] == requirements["led"]["ports"]["drive"] == requirements["reg"]["ports"]["output"]
    buttons = [requirements[f"button{i}"]["ports"]["signal"] for i in (1, 2, 3)]
    assert len(set(buttons)) == 3
    assert [requirements["host"]["ports"][f"pin{i}"] for i in (4, 5, 6)] == buttons
    artifact = lower_requirement(CircuitRequirement.model_validate(control))
    assert artifact is not None
    pins = {(pin.role, pin.pin): pin.net for pin in artifact.pins}
    assert pins[("jumper", "1")] == pins[("pulldown", "2")] == control["ports"]["signal"]
    assert pins[("jumper", "2")] == control["ports"]["vdd"]
    assert pins[("pulldown", "1")] == control["ports"]["gnd"]
    assert any(
        row["kind"] == "quantitative" and row["original_obligation_id"] == "pcb-layer-count"
        and row["value"] == 4
        for row in architecture["obligations"]
    )


def test_physical_attachment_does_not_steal_an_unrelated_block_header():
    from kicraft.server.stage_contracts import attach_uniquely_provable_physical_obligations

    fixture, draft = _retained_1148()
    result = attach_uniquely_provable_physical_obligations(
        draft, {"intent": fixture["intent"], "functional_spec": fixture["functional_spec"]},
    )
    requirements = {row["id"]: row for row in result["requirements"]}
    assert "host-header" in requirements["host"]["obligation_ids"]
    assert "host-header" not in requirements["enable"]["obligation_ids"]


def test_physical_attachment_keeps_ambiguous_same_block_carriers_unselected():
    from kicraft.server.stage_contracts import attach_uniquely_provable_physical_obligations

    fixture, draft = _retained_1148()
    enable = next(row for row in draft["requirements"] if row["id"] == "enable")
    enable["functional_blocks"] = ["HOST_INPUT"]
    other = deepcopy(enable)
    other["id"] = "other"
    other["obligation_ids"] = []
    draft["requirements"].append(other)
    result = attach_uniquely_provable_physical_obligations(
        draft, {"intent": fixture["intent"], "functional_spec": fixture["functional_spec"]},
    )
    requirements = {row["id"]: row for row in result["requirements"]}
    assert "host-header" in requirements["host"]["obligation_ids"]
    assert "host-header" not in requirements["enable"]["obligation_ids"]
    assert "host-header" not in requirements["other"]["obligation_ids"]


def test_switch_input_physical_gate_follows_typed_mechanism():
    from kicraft.server.stage_work_units import _required_physical_feature

    assert _required_physical_feature({
        "family": "switch-input",
        "obligations": [{"kind": "adjustability", "mechanism": "jumper"}],
    }) == "jumper"
    assert _required_physical_feature({"family": "switch-input", "obligations": []}) == "button"
    assert _required_physical_feature({
        "family": "switch-input",
        "obligations": [{"kind": "adjustability", "mechanism": "momentary-button"}],
    }) == "button"
    from kicraft.server.stage_contracts import BomComponentGroup
    from kicraft.server.stage_work_units import _group_has_physical_feature

    button = BomComponentGroup(
        id="button", reference_prefix="SW", quantity=1, value="KMR232NGULCLFS",
        mpn="KMR232NGULCLFS", symbol="kmr232ngulclfs:KMR232NGULCLFS",
        footprint="kmr232ngulclfs:KMR232NGULCLFS", sheet="POWER",
    )
    header = BomComponentGroup(
        id="jumper", reference_prefix="J", quantity=1, value="TSW-102-07-G-S",
        mpn="TSW-102-07-G-S", symbol="Connector_Generic:Conn_01x02",
        footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical", sheet="POWER",
    )
    assert _group_has_physical_feature(header, "jumper")
    assert not _group_has_physical_feature(button, "jumper")
    assert _group_has_physical_feature(button, "button")
    assert not _group_has_physical_feature(header, "button")


def test_retained_draft_builds_all_native_bom_and_wiring():
    from kicraft.server.stage_contracts import _normalize_stage_response
    from kicraft.server.stage_work_units import probe_architecture_construction

    fixture, draft = _retained_1148()
    state = {"intent": fixture["intent"], "functional_spec": fixture["functional_spec"]}
    state["architecture"], _ = _normalize_stage_response("architecture", draft, state)
    constructed = probe_architecture_construction(state)
    assert constructed is not None
    parts = constructed["parts"]
    converter = next(row for row in parts if row.get("recipe_role") == "converter")
    jumper = next(row for row in parts if row.get("lowering_requirement_id") == "enable"
                  and row.get("lowering_role") == "jumper")
    pulldown = next(row for row in parts if row.get("lowering_requirement_id") == "enable"
                    and row.get("lowering_role") == "pulldown")
    host = next(row for row in parts if row.get("lowering_requirement_id") == "host")
    assert jumper["mpn"] == "TSW-102-07-G-S"
    assert any(row["mpn"] == "SNT-100-BL-G" for row in jumper["accessories"])
    assert pulldown["value"] == "10k"
    nets = {
        (endpoint["ref"], endpoint["pin"]): connection["net_name"]
        for connection in constructed["connections"]
        for endpoint in connection["endpoints"]
    }
    assert nets[(converter["ref"], "1")] == nets[(jumper["ref"], "1")] == nets[(pulldown["ref"], "2")]
    assert nets[(jumper["ref"], "2")] == nets[(converter["ref"], "4")] == nets[(host["ref"], "1")] == "+5V"
    assert nets[(pulldown["ref"], "1")] == nets[(host["ref"], "2")] == "GND"
    assert nets[(host["ref"], "3")] == "+3V3"
    for index in (1, 2, 3):
        switch = next(row for row in parts if row.get("lowering_requirement_id") == f"button{index}"
                      and row.get("lowering_role") == "switch")
        assert nets[(switch["ref"], "1")] == nets[(host["ref"], str(index + 3))] == f"BUTTON_{index}"
        assert nets[(switch["ref"], "2")] == "GND"
    led_resistor = next(row for row in parts if row.get("lowering_requirement_id") == "led"
                        and row.get("lowering_role") == "resistor")
    assert nets[(led_resistor["ref"], "1")] == "+3V3"


def _vin_named_draft(voltage):
    fixture, draft = _retained_1148()
    rail = draft["power"]["rails"].pop("+5V")
    rail["voltage"] = voltage
    draft["power"]["rails"]["VIN"] = rail
    next(row for row in draft["requirements"] if row["id"] == "reg")["supply"] = "VIN"
    return fixture, draft


def test_vin_five_volt_control_is_proven_by_sourced_rail_through_native_construction():
    from kicraft.server.stage_contracts import _normalize_stage_response
    from kicraft.server.stage_work_units import probe_architecture_construction

    fixture, draft = _vin_named_draft(5.0)
    state = {"intent": fixture["intent"], "functional_spec": fixture["functional_spec"]}
    state["architecture"], _ = _normalize_stage_response("architecture", draft, state)
    enable = next(row for row in state["architecture"]["requirements"] if row["id"] == "enable")
    assert enable["ports"]["vdd"] == "VIN"
    assert enable["parameters"]["voltage"] == 5.0
    assert probe_architecture_construction(state) is not None


@pytest.mark.parametrize("voltage", [None, 6.0, 24.0])
def test_vin_unknown_or_high_voltage_does_not_produce_enable_circuit(voltage):
    from kicraft.server.stage_contracts import _normalize_stage_response

    fixture, draft = _vin_named_draft(voltage)
    with pytest.raises(ValueError):
        _normalize_stage_response(
            "architecture", draft,
            {"intent": fixture["intent"], "functional_spec": fixture["functional_spec"]},
        )


def test_claimed_safe_voltage_cannot_override_actual_sourced_rail():
    from kicraft.server.stage_contracts import _normalize_stage_response

    fixture, draft = _vin_named_draft(24.0)
    enable = next(row for row in draft["requirements"] if row["id"] == "enable")
    enable.update(
        family="switch-input", supply="VIN", declared_ports=[],
        parameters={"pull_policy": "external", "active_level": "high", "voltage": 5.0},
    )
    next(row for row in draft["signals"] if row["name"] == "REG_ENABLE")["from"] = "enable.signal"
    with pytest.raises(ValueError, match="voltage must match its sourced control rail"):
        _normalize_stage_response(
            "architecture", draft,
            {"intent": fixture["intent"], "functional_spec": fixture["functional_spec"]},
        )


def test_bounded_voltage_does_not_override_explicit_high_voltage_net_name():
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.models import CircuitRequirement

    requirement = CircuitRequirement(
        id="enable", sheet="POWER", role="user_io", family="switch-input",
        exact_part="TSW-102-07-G-S",
        parameters={"pull_policy": "external", "active_level": "high", "voltage": 5.0},
        ports={"signal": "EN", "gnd": "GND", "vdd": "VIN24"},
        obligations=[{"kind": "adjustability", "original_obligation_id": "enable",
                      "parameter": "enable", "mechanism": "jumper"}],
    )
    with pytest.raises(ValueError, match="unbounded raw supply"):
        lower_requirement(requirement)
