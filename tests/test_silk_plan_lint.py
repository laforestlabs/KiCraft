"""Deterministic lint for LLM-authored silkscreen labels.

The lint is the trust boundary between the LLM's content and the physical
board: anchors must exist in the BOM and every electrical claim must be
corroborated by the design state. These tests pin that contract.
"""

import pytest

from kicraft.design.models import (
    Architecture,
    BOM,
    BomPart,
    ConversationState,
    IntentSlot,
    NetConnection,
    PinEndpoint,
    Sheet,
)
from kicraft.design.synthesis.silk_plan import (
    build_corroboration_corpus,
    lint_labels,
    normalize_ascii,
    uncovered_connectors,
)


def _state() -> ConversationState:
    return ConversationState(
        project_stem="PD_TEST",
        intent=IntentSlot(goal="USB-C PD trigger, selectable 9 V / 12 V / 20 V"),
        architecture=Architecture(
            rail_voltages={"VOUT": 12.0},
            sheets=[Sheet(name="PD", stem="PD", function="pd trigger")],
            power_nets=["VBUS", "VOUT"],
            inter_sheet_nets=[],
        ),
        bom=BOM(parts=[
            BomPart(ref="U1", value="CH224K", symbol="ch224k:CH224K",
                    footprint="ch224k:ESSOP-10", sheet="PD"),
            BomPart(ref="SW1", value="DIP-3",
                    symbol="dip-switch-3pos:DSHP03TSGER",
                    footprint="dip-switch-3pos:SW-SMD", sheet="PD"),
            BomPart(ref="J2", value="OUT", symbol="Connector:Conn_01x02",
                    footprint="lib:fp", sheet="PD"),
            BomPart(ref="J1", value="IN", symbol="Connector_Generic:Conn_01x03",
                    footprint="Connector_Generic:Conn_01x03", sheet="PD"),
        ]),
    )


def _lint(labels):
    return lint_labels(labels, _state(), project_root=None)


def test_corpus_includes_rails_and_intent_text():
    corpus = build_corroboration_corpus(_state())
    assert ("12", "V") in corpus  # rail_voltages float
    assert ("9", "V") in corpus  # intent text "9 V"
    assert ("20", "V") in corpus


def test_unknown_anchor_ref_dropped():
    kept, dropped = _lint([
        {"id": "x", "text": "OUT 12V", "anchor": {"ref": "Z99"}},
    ])
    assert kept == []
    assert any("Z99" in d for d in dropped)


def test_uncorroborated_claim_dropped_corroborated_kept():
    kept, dropped = _lint([
        {"id": "amps", "text": "OUT 5A MAX", "anchor": {"ref": "J2"}},
        {"id": "volts", "text": "OUT 9/12/20V", "anchor": {"ref": "J2"}},
    ])
    assert kept == []
    assert any("5A" in d for d in dropped)
    assert any("9V" in d for d in dropped)

def test_slash_voltage_list_is_not_backed_by_unrelated_intent():
    kept, dropped = _lint([
        {"id": "v", "text": "9/12/20V OUT", "anchor": {"ref": "J2"}},
    ])
    assert kept == []
    assert any("uncorroborated" in d for d in dropped)


def test_table_position_guard_uses_switch_pin_count(monkeypatch):
    import kicraft.design.synthesis.silk_plan as silk
    monkeypatch.setattr(silk, "reviewed_interface_facts", lambda *args: [
        {"ref": "SW1", "lines": ["9V 12V 20V"], "pins": []},
    ])
    # dip-switch-3pos is a vendored bundle: 6 pins -> 3 positions. A table
    # header naming position 4 is impossible on this switch.
    kept, dropped = _lint([
        {"id": "tbl", "kind": "table", "anchor": {"ref": "SW1"},
         "text": "VOUT 1 2 3 4\n9V ON - - -"},
    ])
    assert kept == []
    assert any("position 4" in d for d in dropped)

    kept, dropped = _lint([
        {"id": "tbl", "kind": "table", "anchor": {"ref": "SW1"},
         "text": "VOUT 1 2 3\n9V ON - -\n12V - ON -\n20V - - ON"},
    ])
    assert [lb.id for lb in kept] == ["tbl"]


def test_ascii_normalization_and_caps():
    assert normalize_ascii("12 µF · ±5%") == "12 uF - +/-5%"
    long_line = "X" * 60
    kept, _ = _lint([
        {"id": "n", "text": "\n".join([long_line] * 9), "anchor": {"ref": "J2"}},
    ])
    assert len(kept) == 1
    lines = kept[0].text.split("\n")
    assert len(lines) <= 5
    assert all(len(ln) <= 30 for ln in lines)


def test_empty_and_duplicate_labels_dropped():
    kept, dropped = _lint([
        {"id": "a", "text": "   ", "anchor": {"ref": "J2"}},
        {"id": "b", "text": "NOTE", "anchor": {"ref": "J2"}},
        {"id": "b", "text": "NOTE 2", "anchor": {"ref": "J2"}},
    ])
    assert [lb.id for lb in kept] == ["b"]
    assert any("empty" in d for d in dropped)
    assert any("duplicate" in d for d in dropped)


def test_label_cap_keeps_highest_priority():
    # 12 labels: 5 priority-3, 7 priority-1. The cap (10) drops only the two
    # lowest-priority labels; every must-have priority-1 label survives.
    labels = [
        {"id": f"l{i}", "text": "NOTE", "anchor": {"ref": "J2"},
         "priority": 3 if i < 5 else 1}
        for i in range(12)
    ]
    kept, dropped = _lint(labels)
    assert len(kept) == 10
    assert all(any(lb.id == f"l{i}" for lb in kept) for i in range(5, 12))
    assert len([d for d in dropped if "cap" in d]) == 2


def test_pinout_valid_pins_kept():
    kept, dropped = _lint([
        {"id": "j1-pins", "kind": "pinout", "anchor": {"ref": "J1"},
         "pins": [{"pin": "1", "text": "VIN"}, {"pin": "2", "text": "GND"}]},
    ])
    assert dropped == []
    assert [lb.id for lb in kept] == ["j1-pins"]
    assert kept[0].kind == "pinout"
    assert [(p.pin, p.text) for p in kept[0].pins] == [("1", "VIN"), ("2", "GND")]


def test_pinout_unknown_pin_dropped():
    kept, dropped = _lint([
        {"id": "j1-pins", "kind": "pinout", "anchor": {"ref": "J1"},
         "pins": [{"pin": "1", "text": "VIN"}, {"pin": "4", "text": "GND"}]},
    ])
    assert [lb.id for lb in kept] == ["j1-pins"]
    assert [p.pin for p in kept[0].pins] == ["1"]
    assert any("j1-pins:4" in d and "not in symbol" in d for d in dropped)


def test_pinout_uncorroborated_claim_dropped():
    # 48V is not in the _state() corpus (9/12/20 V only); the entry drops.
    kept, dropped = _lint([
        {"id": "p", "kind": "pinout", "anchor": {"ref": "J1"},
         "pins": [{"pin": "1", "text": "48V"}, {"pin": "2", "text": "GND"}]},
    ])
    assert [lb.id for lb in kept] == ["p"]
    assert [e.pin for e in kept[0].pins] == ["2"]
    assert any("p:1" in d and "48V" in d for d in dropped)


def test_pinout_empty_pins_dropped():
    kept, dropped = _lint([
        {"id": "p", "kind": "pinout", "anchor": {"ref": "J1"}, "pins": []},
    ])
    assert kept == []
    assert any("no pins" in d for d in dropped)


def test_pinout_text_truncated_to_max_pin_chars():
    kept, dropped = _lint([
        {"id": "p", "kind": "pinout", "anchor": {"ref": "J1"},
         "pins": [{"pin": "1", "text": "CONNECTOR"}]},
    ])
    assert [lb.id for lb in kept] == ["p"]
    assert kept[0].pins[0].text == "CONNECTO"  # 9 -> 8 chars


def test_uncovered_connectors_reports_unlabeled_wired_connector():
    state = _state()
    state.bom.connections = [
        NetConnection(net_name="VIN", sheet="PD",
                      endpoints=[PinEndpoint(ref="J1", pin="1")]),
    ]
    # No kept label anchors to J1 -> it is reported. J2 is also a connector
    # but is not wired, so it is not reported.
    assert uncovered_connectors([], state) == ["J1"]


def test_interface_projection_preserves_late_connector_and_control_facts(monkeypatch):
    from kicraft.design.synthesis.silk_plan import build_interface_digest
    import kicraft.design.synthesis.electrical_review as review

    state = _state()
    state.architecture.protected_identities = ["duplicated prose " * 20000]
    state.bom.connections = [
        NetConnection(net_name="VBUS", sheet="PD",
                      endpoints=[PinEndpoint(ref="J1", pin="1")]),
        NetConnection(net_name="VOUT", sheet="PD",
                      endpoints=[PinEndpoint(ref="J2", pin="1")]),
        NetConnection(net_name="CFG", sheet="PD",
                      endpoints=[PinEndpoint(ref="SW1", pin="1")]),
    ]
    monkeypatch.setattr(review, "_pin_names", lambda symbol, root: {"1": "CONTACT"})
    digest = build_interface_digest(state)
    assert "duplicated prose" not in digest
    for ref in ("J1", "J2", "SW1"):
        assert f"INTERFACE {ref} — COMPLETE" in digest
    assert '"net":"VOUT"' in digest
    assert '"function":"CONTACT"' in digest


def test_five_unlabeled_connectors_are_defects():
    from kicraft.design.synthesis.silk_plan import required_silk_defects
    state = _state()
    for i in range(3, 6):
        state.bom.parts.append(BomPart(ref=f"J{i}", value="UNKNOWN",
                                      symbol="Connector:Conn_01x02",
                                      footprint="lib:fp", sheet="PD"))
    state.bom.connections = [
        NetConnection(net_name=f"N{i}", sheet="PD",
                      endpoints=[PinEndpoint(ref=f"J{i}", pin="1")])
        for i in range(1, 6)
    ]
    assert len(required_silk_defects(state)) == 5


def test_numeric_claim_is_bound_to_interface(monkeypatch):
    import kicraft.design.synthesis.silk_plan as silk
    monkeypatch.setattr(silk, "reviewed_interface_facts", lambda *args: [
        {"ref": "J1", "lines": ["9V INPUT"], "pins": []},
        {"ref": "J2", "lines": ["5V OUTPUT"], "pins": []},
    ])
    kept, dropped = _lint([{"id": "bad", "text": "9V OUTPUT",
                            "anchor": {"ref": "J2"}}])
    assert not kept
    assert any("9V" in item for item in dropped)


def _reviewed_interfaces_state():
    from kicraft.design.models import RecipeSelection, SilkPlan
    from kicraft.design.recipes import expand_selections
    from kicraft.design.recipes.registry import get_recipe

    recipes = [
        ("ch224k-pd-trigger@2", "pd"),
        ("usb-c-usb2-device@1", "usb"),
        ("hub75-sn74hct245-interface@2", "display"),
        ("ws2812-external-strip-header-interface@1", "led"),
        ("pam8302a-pwm-speaker@1", "audio"),
        ("esp32-s3-wroom-1-n16r8-minimal@1", "mcu"),
        ("ap63205-5v@2", "reg"),
    ]
    selections = []
    for recipe, instance in recipes:
        definition = get_recipe(recipe)
        bindings = {port.name: f"{instance}_{port.name}" for port in definition.ports
                    if port.required}
        bindings.update({port.name: "GND" for port in definition.ports if port.name == "gnd"})
        if "vdd_5v" in bindings:
            bindings["vdd_5v"] = "LED_5V"
        if recipe == "ap63205-5v@2":
            bindings["output"] = "LED_5V"
            bindings["enable"] = "REG_ENABLE"
        if recipe in {"usb-c-usb2-device@1", "esp32-s3-wroom-1-n16r8-minimal@1"}:
            bindings.update(usb_dm="USB_DM", usb_dp="USB_DP")
        parameters = ({"series_resistors": False} if recipe == "usb-c-usb2-device@1"
                      else {"native_usb": True} if recipe.startswith("esp32-s3-wroom-1-")
                      else {})
        selections.append(RecipeSelection(recipe=recipe, instance=instance,
            sheets={role: instance.upper() for role in definition.required_sheet_roles},
            parameters=parameters, port_bindings=bindings))
    expansions = expand_selections(selections)
    state = ConversationState(
        architecture=Architecture(
            sheets=[Sheet(name=instance.upper(), stem=instance.upper(), function=instance)
                    for _, instance in recipes],
            power_nets=[], inter_sheet_nets=[],
                                  recipe_selections=[e.selection for e in expansions]),
        bom=BOM(parts=[p for e in expansions for p in e.parts],
                connections=[c for e in expansions for c in e.connections],
                no_connect_pins=[p for e in expansions for p in e.no_connect_pins],
                recipe_ownership=[e.ownership for e in expansions]),
        silk_plan=SilkPlan(labels=[]))
    return state


def test_required_silk_lifecycle_survives_disabled_and_failed_provider(tmp_path, monkeypatch):
    from kicraft.design.cli_app import run_silk_plan_authoring
    import kicraft.design.cli_app as cli
    from kicraft.design.synthesis.interface_facts import reviewed_interface_facts

    state = _reviewed_interfaces_state()
    path = tmp_path / "state.json"
    path.write_text(state.model_dump_json())
    monkeypatch.setenv("KICRAFT_SILK_PLAN", "0")
    calls = []
    def failed_provider(*args):
        calls.append(True)
        raise RuntimeError("optional provider unavailable")
    monkeypatch.setattr(cli, "_review_client_and_settings", failed_provider)
    run_silk_plan_authoring(path, tmp_path, lambda event: None)
    assert not calls
    saved = ConversationState.model_validate_json(path.read_text())
    facts = reviewed_interface_facts(saved)
    assert {f["kind"] for f in facts} == {
        "pd_power", "usb_programming", "hub75", "led_output", "speaker", "boot", "reset"}
    assert next(f for f in facts if f["kind"] == "led_output")["current_a"] == 0.25
    assert next(f for f in facts if f["kind"] == "speaker")["impedance_ohm"] == 8
    assert saved.stage_status["silk_plan"].semantic_clean
    before = saved.silk_plan.labels
    monkeypatch.setenv("KICRAFT_SILK_PLAN", "1")
    run_silk_plan_authoring(path, tmp_path, lambda event: None)
    assert calls == [True]
    saved = ConversationState.model_validate_json(path.read_text())
    assert saved.silk_plan.labels == before
    assert saved.stage_status["silk_plan"].semantic_clean


def test_required_pin_placement_receipt_is_checked():
    from kicraft.design.models import SilkPlan
    from kicraft.design.synthesis.silk_plan import (
        required_labels, required_silk_defects,
    )
    state = _reviewed_interfaces_state()
    state.silk_plan = SilkPlan(labels=required_labels(state))
    assert not required_silk_defects(state)
    defects = required_silk_defects(state, placed_ids=[])
    assert any(":pins:1" in defect and "not placed" in defect for defect in defects)


def test_changed_led_owned_wiring_does_not_produce_numeric_fact():
    from kicraft.design.synthesis.interface_facts import reviewed_interface_facts
    state = _reviewed_interfaces_state()
    led = next(p for p in state.bom.parts
               if p.recipe_instance == "led" and p.recipe_role == "connector")
    net = next(c for c in state.bom.connections
               if any(ep.ref == led.ref and ep.pin == "2" for ep in c.endpoints))
    net.net_name = "UNREVIEWED_DATA"
    assert not any(f["kind"] == "led_output" for f in reviewed_interface_facts(state))


def test_recipe_and_lowered_interfaces_supply_required_physical_labels():
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.models import CircuitRequirement, RecipeSelection, SilkPlan
    from kicraft.design.recipes import expand_selections
    from kicraft.design.synthesis.silk_plan import required_labels, required_silk_defects

    expansions = expand_selections([
        RecipeSelection(recipe="ch32v003f4u6-minimal@1", instance="core",
                        sheets={"mcu": "BOARD"}, port_bindings={"vdd": "VDD", "gnd": "GND"}),
        RecipeSelection(recipe="usb-c-5v-sink@1", instance="usb",
                        sheets={"power": "BOARD"}, port_bindings={"vbus": "VBUS", "gnd": "GND"}),
    ])
    connector = CircuitRequirement(
        id="aux", sheet="BOARD", role="connector", family="qwiic-i2c-connector",
        ports={"vcc": "VDD", "ground": "GND", "sda": "BOARD_SDA", "scl": "BOARD_SCL"},
    )
    artifact = lower_requirement(connector)
    group, = artifact.groups
    state = ConversationState(
        architecture=Architecture(
            sheets=[Sheet(name="BOARD", stem="BOARD", function="interfaces")],
            requirements=[connector], recipe_selections=[row.selection for row in expansions],
            rail_voltages={"VDD": 3.3, "VBUS": 5.0, "GND": 0},
            power_nets=["VDD", "VBUS", "GND"], inter_sheet_nets=[],
        ),
        bom=BOM(
            parts=[*[part for row in expansions for part in row.parts],
                   BomPart(ref="J30", sheet="BOARD", value=group.value, symbol=group.symbol,
                           footprint=group.footprint, mpn=group.mpn, resolution_source="lowerer",
                           resolution_id=artifact.lowerer_id, lowering_role=group.role,
                           lowering_index=0, lowering_requirement_id=connector.id)],
            connections=[*[net for row in expansions for net in row.connections],
                         *[NetConnection(net_name=pin.net, sheet="BOARD",
                                         endpoints=[PinEndpoint(ref="J30", pin=pin.pin)])
                           for pin in artifact.pins]],
            no_connect_pins=[pin for row in expansions for pin in row.no_connect_pins],
            recipe_ownership=[row.ownership for row in expansions],
        ),
    )
    state.silk_plan = SilkPlan(labels=required_labels(state))
    assert {label.text for label in state.silk_plan.labels if label.kind == "io"} == {
        "RESET", "WCH-LINK", "USB 5V IN", "I2C / QWIIC",
    }
    assert not required_silk_defects(state)
    assert "required silk required:J30:qwiic:pins:3: not placed" in required_silk_defects(
        state, placed_ids=[]
    )
    changed = next(net for net in state.bom.connections
                   if any(ep.ref == "J30" and ep.pin == "3" for ep in net.endpoints))
    changed.net_name = "UNRELATED_SIGNAL"
    assert "required silk J30: interface function not verified" in required_silk_defects(state)


@pytest.mark.parametrize("recipe", ["rp2040-minimal@4", "rp2040-minimal@5"])
def test_rp2040_swd_silk_requires_actual_run_and_bootsel_circuits(recipe):
    from kicraft.design.models import RecipeSelection, SilkPlan
    from kicraft.design.recipes import expand_recipe
    from kicraft.design.synthesis.interface_facts import reviewed_interface_facts
    from kicraft.design.synthesis.silk_plan import required_labels, required_silk_defects

    selection = RecipeSelection(
        recipe=recipe, instance="rp2040", sheets={"mcu": "BOARD", "io": "BOARD"},
        parameters={"native_usb": False, "programming": "swd"},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
    )
    expanded = expand_recipe(selection)
    state = ConversationState(
        architecture=Architecture(
            sheets=[Sheet(name="BOARD", stem="BOARD", function="RP2040 controller")],
            recipe_selections=[expanded.selection], rail_voltages={"+3V3": 3.3, "GND": 0},
            power_nets=["+3V3", "GND"], inter_sheet_nets=[],
        ),
        bom=BOM(parts=expanded.parts, connections=expanded.connections,
                no_connect_pins=expanded.no_connect_pins, recipe_ownership=[expanded.ownership]),
    )
    facts = {row["title"]: row for row in reviewed_interface_facts(state)}
    assert set(facts) == {"BOOTSEL", "RESET", "SWD"}
    assert [(pin["pin"], pin["text"]) for pin in facts["SWD"]["pins"]] == [
        ("1", "VTREF"), ("2", "GND"), ("3", "SWDIO"), ("4", "SWCLK"), ("5", "RUN"),
    ]
    state.silk_plan = SilkPlan(labels=required_labels(state))
    assert not required_silk_defects(state)

    reset_ref = facts["RESET"]["ref"]
    run = next(net for net in state.bom.connections
               if any((ep.ref, str(ep.pin)) == (reset_ref, "1") for ep in net.endpoints))
    run.endpoints = [ep for ep in run.endpoints if ep.ref != reset_ref]
    defects = required_silk_defects(state)
    assert f"required silk {reset_ref}: control function not verified" in defects
    assert f"required silk {facts['SWD']['ref']}: interface function not verified" in defects
