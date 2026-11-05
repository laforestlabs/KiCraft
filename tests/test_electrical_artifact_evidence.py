from types import SimpleNamespace

import pytest

from kicraft.design.part_identity import reviewed_part
from kicraft.design.models import RecipeSelection
from kicraft.design.recipes import expand_recipe
from kicraft.design.synthesis.symbol_pinout import lookup_pins
from kicraft.eval import electrical_artifact_evidence
from kicraft.eval.design_acceptance import evaluate_obligation


class _Pad:
    def __init__(self, number, net):
        self._number = number
        self._net = net

    def GetNumber(self):
        return self._number

    def GetNetname(self):
        return self._net


class _Footprint:
    def __init__(self, ref, footprint, pads):
        self._ref = ref
        self._footprint = footprint
        self._pads = [_Pad(*pad) for pad in pads]

    def GetReferenceAsString(self):
        return self._ref

    def GetFPID(self):
        return SimpleNamespace(
            GetLibItemName=lambda: self._footprint.rsplit(":", 1)[-1],
            GetLibNickname=lambda: self._footprint.rpartition(":")[0],
        )

    def Pads(self):
        return self._pads


class _Board:
    def __init__(self, footprints):
        self._footprints = footprints

    def GetFootprints(self):
        return self._footprints


def _output_obligation(voltage):
    return {
        "id": f"output-{voltage}",
        "check": {
            "kind": "numeric_range", "fact": "output_voltage_v",
            "minimum": voltage, "maximum": voltage,
        },
    }


def _contract(voltage):
    return {"obligations": [_output_obligation(voltage)]}


def _obligation_result(tmp_path, obligation, facts):
    artifact = tmp_path / "facts.json"
    artifact.write_text("{}")
    return evaluate_obligation(
        obligation,
        {
            "status": "pass",
            "evidence": [{"kind": "tool-report", "path": artifact.name}],
            "facts": facts,
        },
        artifact_root=tmp_path,
    )


def _buck_artifact(recipes, *, defect=None, equal_feedback=False, input_voltage=12.0):
    terminal = reviewed_part("WJ126V-5.0-02P-14-00A")
    rows, parts, names, inputs = [], [], {}, []
    rails = {"GND": 0.0}
    for index, recipe in enumerate(recipes):
        vin = f"VIN{index}"
        output = "+3V3" if equal_feedback else f"OUT{index}"
        rails[vin] = input_voltage
        if equal_feedback:
            rails[output] = 3.3  # Deliberately disagrees with the delivered divider.
        expanded = expand_recipe(RecipeSelection(
            recipe=recipe, instance=f"power{index}", sheets={"power": "POWER"},
            port_bindings={"input": vin, "output": output, "gnd": "GND"},
        ))
        refs = {
            part.ref: f"{part.ref.rstrip('0123456789')}{int(part.ref.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')) + index * 100}"
            for part in expanded.parts
        }
        nets = {
            (refs[endpoint.ref], endpoint.pin): connection.net_name
            for connection in expanded.connections for endpoint in connection.endpoints
        }
        converter_ref = next(
            refs[part.ref] for part in expanded.parts
            if (part.mpn or "").casefold() in {
                "ap63205wu-7", "ap63203wu-7", "tps54331ddar", "tlv62569dbvr",
            }
        )
        branch_parts = []
        for part in expanded.parts:
            row = part.model_dump(mode="json")
            row["ref"] = refs[part.ref]
            branch_parts.append(row)
            pins = lookup_pins(part.symbol, all_units=True)["pins"]
            pads = []
            for pin in pins:
                number, name = str(pin["number"]), str(pin.get("name") or "")
                names[row["ref"], number] = name
                net = nets.get((row["ref"], number), "")
                if defect == "bad_feedback" and index == 0 and row["ref"] == converter_ref and name in {"FB", "VSENSE"}:
                    net = "FB_BAD"
                pads.append((number, net))
            footprint = part.footprint
            if index == 0 and row["ref"] == converter_ref:
                if defect == "missing_converter":
                    continue
                if defect == "corrupt_converter_evidence":
                    footprint = "other-lib:wrong-footprint"
            rows.append((row["ref"], footprint, pads))
        if equal_feedback:
            feedback_pin = next(pin for (ref, pin), name in names.items() if ref == converter_ref and name == "VSENSE")
            feedback = nets[converter_ref, feedback_pin]
            for part in branch_parts:
                endpoints = {net for (ref, _), net in nets.items() if ref == part["ref"]}
                if part["symbol"] == "Device:R" and endpoints in ({output, feedback}, {feedback, "GND"}):
                    part["value"] = "10k"
                    part["mpn"] = "FRC0603F1002TS"
                    part["sourcing_note"] = None
        parts.extend(branch_parts)
        for offset, net in enumerate((vin, output)):
            ref = f"J{900 + 2 * index + offset}"
            parts.append({
                "value": "POWER", "sheet": "POWER",
                "ref": ref, "mpn": terminal.identity, "symbol": terminal.symbol,
                "footprint": terminal.footprint,
            })
            delivered = "LOAD_FLOAT" if defect == "disconnected_output" and index == 0 and offset == 1 else net
            rows.append((ref, terminal.footprint, (("1", delivered), ("2", "GND"))))
        inputs.append({
            "id": f"input{index}", "role": "power_input", "family": "screw-terminal",
            "exact_part": terminal.identity,
            "ports": {"positive": vin, "negative": "GND"},
        })
    state = {"bom": {"parts": parts},
             "architecture": {"rail_voltages": rails, "requirements": inputs}}
    return state, _Board([_Footprint(*row) for row in rows]), names


def _fixed_buck_artifact(*, branches=1, defect=None):
    recipes = ["ap63205-5v@1"] + ["ap63203-3v3@1"] * (branches - 1)
    return _buck_artifact(recipes, defect=defect)


def _adjustable_buck_artifact():
    return _buck_artifact(["tps54331-adjustable@2"], equal_feedback=True)


def test_isolated_converter_cannot_establish_a_board_input_voltage_rating(tmp_path, monkeypatch):
    state, board, names = _fixed_buck_artifact()
    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            pad._net = ""
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    contract = {"obligations": [{
        "id": "input-envelope",
        "check": {"kind": "numeric_range", "fact": "input_voltage_max_v", "minimum": 8.4},
    }]}
    facts = electrical_artifact_evidence.extract_electrical_facts(
        tmp_path, state, board, contract
    )
    assert "input_voltage_max_v" not in facts


def test_wrong_delivered_divider_cannot_fulfil_the_named_rail(tmp_path, monkeypatch):
    state, board, names = _adjustable_buck_artifact()
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, _contract(3.3))

    assert _obligation_result(tmp_path, _output_obligation(3.3), facts) is not None


@pytest.mark.parametrize("defect", [None, "bad_feedback"])
def test_tlv62569_output_requires_its_delivered_adjustable_divider(tmp_path, monkeypatch, defect):
    state, board, names = _buck_artifact(["tlv62569-3v3@3"], defect=defect, input_voltage=5.0)
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(
        tmp_path, state, board, _contract(3.3),
    )
    if defect:
        assert "output_voltage_v" not in facts
    else:
        assert facts["output_voltage_v"] == pytest.approx(0.6 * (1 + 100 / 22))


def test_fixed_output_voltage_requires_delivered_converter_topology(tmp_path, monkeypatch):
    state, board, names = _fixed_buck_artifact()
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, _contract(12))

    assert facts["output_voltage_v"] == pytest.approx(5.0)
    assert _obligation_result(tmp_path, _output_obligation(5), facts) is None
    assert _obligation_result(tmp_path, _output_obligation(12), facts) is not None


@pytest.mark.parametrize("defect", ["disconnected_output", "missing_converter", "corrupt_converter_evidence", "bad_feedback"])
def test_fixed_output_voltage_negative_topologies_remain_unverified(tmp_path, monkeypatch, defect):
    state, board, names = _fixed_buck_artifact(defect=defect)
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, _contract(5))

    assert "output_voltage_v" not in facts
    assert _obligation_result(tmp_path, _output_obligation(5), facts) is not None


def test_multiple_physical_outputs_remain_unverified(tmp_path, monkeypatch):
    state, board, names = _fixed_buck_artifact(branches=2)
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, _contract(5))

    assert "output_voltage_v" not in facts


def test_mcu_artifact_observer_requires_real_stm32_support_and_swd_header_graph():
    from kicraft.eval.mcu_artifact_evidence import mcu_circuit_facts

    stm = reviewed_part("stm32g030f6p6")
    parts = [
        {"ref": "U1", "mpn": stm.identity, "symbol": stm.symbol, "footprint": stm.footprint},
        {"ref": "C1", "value": "100nF"}, {"ref": "C2", "value": "100nF"},
        {"ref": "R1", "value": "10k"}, {"ref": "R2", "value": "10k"},
        {"ref": "SW1"}, {"ref": "J1"},
    ]
    nets = {
        ("U1", "4"): "VDD", ("U1", "5"): "GND", ("U1", "6"): "NRST",
        ("U1", "18"): "SWDIO", ("U1", "19"): "SWCLK",
        ("C1", "1"): "VDD", ("C1", "2"): "GND",
        ("C2", "1"): "NRST", ("C2", "2"): "GND",
        ("R1", "1"): "NRST", ("R1", "2"): "VDD",
        ("R2", "1"): "SWCLK", ("R2", "2"): "GND",
        ("SW1", "1"): "NRST", ("SW1", "2"): "GND",
        ("J1", "1"): "VDD", ("J1", "2"): "GND", ("J1", "3"): "SWDIO",
        ("J1", "4"): "SWCLK", ("J1", "5"): "NRST",
    }
    assert mcu_circuit_facts(parts, {}, nets)["net_paths"]["stm32g030f6p6_complete_core"]
    del nets["J1", "5"]
    assert "stm32g030f6p6_complete_core" not in mcu_circuit_facts(parts, {}, nets)["net_paths"]


def test_rp2040_swd_observer_requires_delivered_recovery_and_core_graph():
    from kicraft.eval.mcu_artifact_evidence import mcu_circuit_facts

    expansion = expand_recipe(
        RecipeSelection(
            recipe="rp2040-minimal@3",
            instance="rp",
            sheets={"mcu": "MCU", "io": "IO"},
            parameters={"native_usb": False, "programming": "swd"},
            port_bindings={
                "vdd": "+3V3",
                "gnd": "GND",
                "uart_tx": "MCU_TO_HOST",
                "uart_rx": "HOST_TO_MCU",
            },
        )
    )
    parts = [part.model_dump() for part in expansion.parts]
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }

    facts = mcu_circuit_facts(parts, {}, nets)

    assert facts["net_paths"]["rp2040_swd_recovery_complete_core"]
    mcu = next(part for part in expansion.parts if part.recipe_role == "mcu")
    flash = next(part for part in expansion.parts if part.recipe_role == "flash")
    for mcu_pin, flash_pin in (("52", "6"), ("53", "5"), ("55", "2"), ("54", "3"), ("51", "7")):
        broken = dict(nets)
        del broken[mcu.ref, mcu_pin]
        del broken[flash.ref, flash_pin]
        assert "rp2040_swd_recovery_complete_core" not in mcu_circuit_facts(
            parts, {}, broken
        )["net_paths"]
    header = next(part for part in expansion.parts if part.recipe_role == "swd_header")
    del nets[header.ref, "3"]
    assert "rp2040_swd_recovery_complete_core" not in mcu_circuit_facts(
        parts, {}, nets
    )["net_paths"]

def _usb_bridge_artifact(defect=None):
    specs = [
        (
            "ap63205-5v@1" if defect == "five_volt_logic" else "ap63203-3v3@1",
            "power", {"power": "POWER"},
            {"input": "VBUS", "output": "+3V3", "gnd": "GND"},
        ),
        (
            "cp2102n-usb-uart@1", "bridge", {"interface": "USB"},
            {"vbus": "VBUS", "vdd": "+3V3", "gnd": "GND", "usb_dp": "USB_DP",
             "usb_dm": "USB_DM", "tx": "HOST_TO_MCU", "rx": "MCU_TO_HOST"},
        ),
        (
            "stm32g030f6p6-minimal@2", "mcu", {"mcu": "MCU"},
            {"vdd": "+3V3", "gnd": "GND", "uart_rx": "HOST_TO_MCU", "uart_tx": "MCU_TO_HOST"},
        ),
    ]
    parts, connections, no_connects = [], [], []
    for index, (recipe, instance, sheets, ports) in enumerate(specs):
        expansion = expand_recipe(RecipeSelection(
            recipe=recipe, instance=instance, sheets=sheets, port_bindings=ports,
        ))
        refs = {
            part.ref: f"{part.ref.rstrip('0123456789')}{int(part.ref.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')) + 100 * index}"
            for part in expansion.parts
        }
        parts.extend(part.model_dump(mode="json") | {"ref": refs[part.ref]} for part in expansion.parts)
        connections.extend(
            connection.model_dump(mode="json") | {
                "endpoints": [{"ref": refs[endpoint.ref], "pin": endpoint.pin} for endpoint in connection.endpoints],
            }
            for connection in expansion.connections
        )
        no_connects.extend(
            endpoint.model_dump(mode="json") | {"ref": refs[endpoint.ref]}
            for endpoint in expansion.no_connect_pins
        )
    nets = {
        (endpoint["ref"], endpoint["pin"]): connection["net_name"]
        for connection in connections for endpoint in connection["endpoints"]
    }
    role_refs = {(part["recipe_instance"], part["recipe_role"]): part["ref"] for part in parts}
    bridge = role_refs["bridge", "bridge"]
    mcu = role_refs["mcu", "mcu"]
    usb = role_refs["bridge", "connector"]
    esd = role_refs["bridge", "esd"]
    defects = {
        "regin_to_usb": {(bridge, "7"): "VBUS"},
        "sense_to_usb": {(bridge, "8"): "VBUS"},
        "missing_upper_divider": {(role_refs["bridge", "vbus_sense_upper"], "1"): "OPEN"},
        "missing_lower_divider": {(role_refs["bridge", "vbus_sense_lower"], "2"): "OPEN"},
        "missing_bridge_reset_pullup": {(role_refs["bridge", "reset_pullup"], "1"): "OPEN"},
        "exposed_ground_open": {(bridge, "29"): "OPEN"},
        "usb_ground_open": {(usb, "A1B12"): "OPEN"},
        "usb_data_open": {(usb, "A6"): "OPEN"},
        "cc_contacts_shorted": {(usb, "B5"): nets[usb, "A5"]},
        "esd_supply_open": {(esd, "5"): "OPEN"},
        "uart_not_crossed": {(mcu, "9"): "HOST_TO_MCU", (mcu, "10"): "MCU_TO_HOST"},
        "reset_button_open": {(role_refs["mcu", "reset_button"], "1"): "OPEN"},
        "reset_held_low_supply_short": {
            (mcu, "6"): "GND", (role_refs["mcu", "reset_button"], "1"): "+3V3",
        },
    }
    nets.update(defects.get(defect, {}))
    footprints = [
        _Footprint(
            part["ref"], part["footprint"],
            [(str(pin["number"]), nets.get((part["ref"], str(pin["number"])), ""))
             for pin in lookup_pins(part["symbol"], all_units=True)["pins"]],
        )
        for part in parts
    ]
    state = {
        "bom": {"parts": parts, "connections": connections, "no_connect_pins": no_connects},
        "architecture": {"rail_voltages": {"VBUS": 5.0, "+3V3": 3.3, "GND": 0.0}},
    }
    return state, _Board(footprints)


@pytest.mark.parametrize("defect", [
    None, "five_volt_logic", "regin_to_usb", "sense_to_usb",
    "missing_upper_divider", "missing_lower_divider", "missing_bridge_reset_pullup",
    "exposed_ground_open", "usb_ground_open", "usb_data_open",
    "cc_contacts_shorted", "esd_supply_open", "uart_not_crossed",
])
def test_usb_bridge_requires_safe_power_sense_and_complete_physical_interfaces(tmp_path, defect):
    state, board = _usb_bridge_artifact(defect)
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, {})
    assert facts["net_paths"].get("usb_uart_bridge", False) is (defect is None)


def _complete_controller_screw_terminal_artifact(fault=None):
    state, board = _usb_bridge_artifact(
        "five_volt_logic" if fault == "unpowered" else None
    )
    terminal = reviewed_part("WJ126V-5.0-02P-14-00A")
    parts = state["bom"]["parts"]
    mcu = next(
        part["ref"]
        for part in parts
        if part["recipe_instance"] == "mcu" and part["recipe_role"] == "mcu"
    )
    mcu_footprint = next(footprint for footprint in board._footprints if footprint._ref == mcu)
    terminal_pads = [
        ("T1", "GPIO_A", "GND"),
        ("T2", "GPIO_B", "OTHER_GND" if fault == "cross_domain_return" else "GND"),
    ]
    if fault == "rail_only":
        terminal_pads = [("T1", "+3V3", "GND"), ("T2", "+3V3", "GND")]
    for index, (ref, signal, ground) in enumerate(terminal_pads, start=1):
        parts.append({
            "ref": ref, "mpn": terminal.identity, "symbol": terminal.symbol,
            "footprint": terminal.footprint,
        })
        board._footprints.append(_Footprint(
            ref, terminal.footprint,
            [("1", "FLOAT" if fault == "disconnected" and ref == "T2" else signal), ("2", ground)],
        ))
        if fault != "rail_only":
            mcu_pin = str(index)
            next(pad for pad in mcu_footprint._pads if pad._number == mcu_pin)._net = signal
    if fault == "cross_domain_return":
        board._footprints.append(_Footprint("R_FAULT", "Resistor_SMD:R_0603_1608Metric", [("1", "OTHER_GND")]))
    return state, board


@pytest.mark.parametrize("fault", [None, "disconnected", "rail_only", "cross_domain_return", "unpowered"])
def test_four_screw_terminal_io_gate_requires_a_powered_complete_controller(tmp_path, fault):
    state, board = _complete_controller_screw_terminal_artifact(fault)
    contract = {"obligations": [{
        "id": "screw-terminal-io",
        "check": {"kind": "gate", "gate": "external_four_screw_terminal_io_contacts"},
    }]}
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, contract)
    assert (facts["gates"].get("external_four_screw_terminal_io_contacts") == "pass") is (fault is None)


def test_programming_evidence_preserves_independently_observed_reset_control(tmp_path):
    state, board = _usb_bridge_artifact()
    contract = {"obligations": [{
        "id": "reset", "check": {"kind": "gate", "gate": "external_reset_control_function"},
    }]}
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, contract)
    assert facts["gates"]["programming"] == "pass"
    assert facts["gates"]["external_reset_control_function"] == "pass"


@pytest.mark.parametrize("defect", ["reset_button_open", "reset_held_low_supply_short"])
def test_disconnected_or_permanently_asserted_reset_is_not_a_working_control(tmp_path, defect):
    state, board = _usb_bridge_artifact(defect)
    contract = {"obligations": [{
        "id": "reset", "check": {"kind": "gate", "gate": "external_reset_control_function"},
    }]}
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, contract)
    assert facts["gates"].get("external_reset_control_function") != "pass"


def _analog_power_cycle_artifact(*, defect=None, rail_voltage=5.0):
    terminal = reviewed_part("WJ126V-5.0-02P-14-00A")
    switch = reviewed_part("TPS22948DCKR")
    supervisor = reviewed_part("TPS3808G33DBVR" if rail_voltage == 3.3 else "TPS3808G50DBVR")
    isolator = reviewed_part("SN74LVC1G66DBVR")
    buffer = reviewed_part("SN74LVC1G17DBVR")
    button = reviewed_part("TL3342F260QG")
    opamp = reviewed_part("MCP6002-I/SN")
    passive = {"symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric"}
    capacitor = {"symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric"}
    parts = [
        {"ref": "U1", "mpn": switch.identity, "symbol": switch.symbol, "footprint": switch.footprint},
        {"ref": "U2", "mpn": opamp.identity, "symbol": opamp.symbol, "footprint": opamp.footprint},
        {"ref": "U3", "mpn": supervisor.identity, "symbol": supervisor.symbol, "footprint": supervisor.footprint},
        {"ref": "U4", "mpn": isolator.identity, "symbol": isolator.symbol, "footprint": isolator.footprint},
        {"ref": "U5", "mpn": buffer.identity, "symbol": buffer.symbol, "footprint": buffer.footprint},
        {"ref": "SW1", "mpn": button.identity, "symbol": button.symbol, "footprint": button.footprint},
        {"ref": "R1", "value": "5.9k", **passive},
        {"ref": "R2", "value": "60k", **passive},
        {"ref": "R4", "value": "1k", **passive},
        {"ref": "R5", "value": "100k", **passive},
        {"ref": "C1", "value": "1uF", **capacitor},
        {"ref": "C2", "value": "100nF", **capacitor},
        {"ref": "C4", "value": "100nF", **capacitor},
        {"ref": "C5", "value": "100nF", **capacitor},
        {"ref": "J1", "mpn": terminal.identity, "symbol": terminal.symbol, "footprint": terminal.footprint},
        {"ref": "J2", "mpn": terminal.identity, "symbol": terminal.symbol, "footprint": terminal.footprint},
    ]
    by_ref = {str(part["ref"]): part for part in parts}
    if defect == "insufficient_input_bypass":
        by_ref["C1"]["value"] = "1nF"
    if defect == "weak_supervisor_pullup":
        by_ref["R5"]["value"] = "1Meg"
    if defect == "legacy_switch":
        legacy = reviewed_part("TPS2553DBVR")
        by_ref["U1"].update({"mpn": legacy.identity, "symbol": legacy.symbol, "footprint": legacy.footprint})
    if defect == "unsafe_leakage":
        by_ref["R4"]["value"] = "10k"
    if defect == "wrong_supervisor_variant":
        wrong = reviewed_part("TPS3808G50DBVR" if rail_voltage == 3.3 else "TPS3808G33DBVR")
        by_ref["U3"].update({"mpn": wrong.identity, "symbol": wrong.symbol, "footprint": wrong.footprint})
    bleed = "FLOAT" if defect == "missing_bleed" else "GND"
    button_ground = "NOT_GND" if defect == "wrong_button" else "GND"
    opamp_supply = "VIN" if defect == "raw_opamp_supply" else "ANALOG_5V"
    supervisor_vdd = "ANALOG_5V" if defect == "supervisor_on_switched_rail" else "VIN"
    supervisor_sense = "VIN" if defect == "supervisor_sense_on_input" else "ANALOG_5V"
    supervisor_reset = "EN" if defect == "supervisor_reset_on_enable" else "RESET_RELEASE"
    control = "RESET_RELEASE" if defect == "unbuffered_isolation_control" else "ISOLATION_ENABLE"
    buffer_vdd = "ANALOG_5V" if defect == "buffer_on_switched_rail" else "VIN"
    buffer_input = "ISOLATION_ENABLE" if defect == "buffer_wrong_polarity" else "RESET_RELEASE"
    buffer_output = "RESET_RELEASE" if defect == "buffer_wrong_polarity" else "ISOLATION_ENABLE"
    supervisor_pads = [
        ("1", supervisor_reset), ("2", "GND"), ("3", "VIN"),
        ("5", supervisor_sense), ("6", supervisor_vdd),
    ]
    if defect == "supervisor_ct_tied":
        supervisor_pads.append(("4", "GND"))
    reset_pullup = "NOT_EN" if defect == "missing_supervisor_pullup" else "EN"
    connector_signal = "OUTA" if defect == "external_signal_backpower" else "HOST_OUTA"
    rows = [
        ("U1", by_ref["U1"]["footprint"], [("1", "VIN"), ("2", "GND"), ("3", "EN"), ("6", "ANALOG_5V")]),
        ("U2", opamp.footprint, [("1", "OUTA"), ("2", "INA-"), ("3", "INA+"), ("4", "GND"), ("5", "INB+"), ("6", "INB-"), ("7", "OUTB"), ("8", opamp_supply)]),
        ("U3", by_ref["U3"]["footprint"], supervisor_pads),
        ("U4", isolator.footprint, [("1", "OUTA"), ("2", "HOST_OUTA"), ("3", "GND"), ("4", control), ("5", "VIN")]),
        ("U5", buffer.footprint, [("2", buffer_input), ("3", "GND"), ("4", buffer_output), ("5", buffer_vdd)]),
        ("SW1", button.footprint, [("1", "EN"), ("2", button_ground)]),
        ("R1", passive["footprint"], [("1", "VIN"), ("2", "EN")]),
        ("R2", passive["footprint"], [("1", "EN"), ("2", "GND")]),
        ("R4", passive["footprint"], [("1", "ANALOG_5V"), ("2", bleed)]),
        ("R5", passive["footprint"], [("1", "RESET_RELEASE"), ("2", reset_pullup)]),
        ("C1", capacitor["footprint"], [("1", "VIN"), ("2", "GND")]),
        ("C2", capacitor["footprint"], [("1", "VIN"), ("2", "GND")]),
        ("C4", capacitor["footprint"], [("1", "VIN"), ("2", "GND")]),
        ("C5", capacitor["footprint"], [("1", "VIN"), ("2", "GND")]),
        ("J1", terminal.footprint, [("1", connector_signal), ("2", "GND")]),
        ("J2", terminal.footprint, [("1", "VIN"), ("2", "GND")]),
    ]
    if defect == "missing_control_buffer":
        parts.remove(by_ref["U5"])
        rows = [row for row in rows if row[0] != "U5"]
    if defect == "buffer_output_capacitor":
        parts.append({"ref": "C6", "value": "10nF", **capacitor})
        rows.append(("C6", capacitor["footprint"], [("1", "ISOLATION_ENABLE"), ("2", "GND")]))
    if defect == "missing_supervisor":
        parts.remove(by_ref["U3"])
        rows = [row for row in rows if row[0] != "U3"]
    if defect in {"excess_switched_capacitance", "unparseable_switched_capacitance"}:
        parts.append({"ref": "C3", "value": "2.2uF" if defect == "excess_switched_capacitance" else "unknown", **capacitor})
        rows.append(("C3", capacitor["footprint"], [("1", "ANALOG_5V"), ("2", "GND")]))
    if defect == "unbounded_passive_load":
        parts.append({"ref": "R6", "value": "unknown", **passive})
        rows.append(("R6", passive["footprint"], [("1", "ANALOG_5V"), ("2", "GND")]))
    if defect == "excess_output_load":
        parts.append({"ref": "R6", "value": "10R", **passive})
        rows.append(("R6", passive["footprint"], [("1", "OUTA"), ("2", "GND")]))
    if defect == "excess_parallel_bleeds":
        for index in range(21):
            ref = f"R{index + 6}"
            parts.append({"ref": ref, "value": "1k", **passive})
            rows.append((ref, passive["footprint"], [("1", "ANALOG_5V"), ("2", "GND")]))
    bypasses = {
        "parallel_power_feed": ("VIN", "ANALOG_5V", "10R"),
        "passive_signal_backpower": ("OUTA", "HOST_OUTA", "100R"),
        "unswitched_signal_bias": ("VIN", "INA+", "100k"),
    }
    if defect in bypasses:
        left, right, resistance = bypasses[defect]
        parts.append({"ref": "R6", "value": resistance, **passive})
        rows.append(("R6", passive["footprint"], [("1", left), ("2", right)]))
    names = {
        ("U1", "1"): "IN", ("U1", "2"): "GND", ("U1", "3"): "ON",
        ("U1", "6"): "OUT",
        ("U2", "4"): "VSS", ("U2", "8"): "VDD",
    }
    state = {
        "bom": {"parts": parts},
        "architecture": {
            "requirements": [{
                "family": "analog-power-cycle-reset",
                "parameters": {"rail_voltage_v": rail_voltage, "source_current_a": 0.5},
                "ports": {"input": "VIN", "output": "ANALOG_5V", "gnd": "GND"},
            }],
        },
    }
    return state, _Board([_Footprint(*row) for row in rows]), names


@pytest.mark.parametrize("defect", [
    None, "missing_bleed", "unsafe_leakage", "wrong_button", "raw_opamp_supply", "insufficient_input_bypass",
    "legacy_switch", "excess_switched_capacitance",
    "unparseable_switched_capacitance", "missing_supervisor", "wrong_supervisor_variant",
    "supervisor_on_switched_rail", "supervisor_sense_on_input",
    "supervisor_reset_on_enable", "supervisor_ct_tied", "missing_supervisor_pullup",
    "external_signal_backpower", "unbounded_passive_load", "excess_output_load",
    "excess_parallel_bleeds", "parallel_power_feed", "passive_signal_backpower",
    "unswitched_signal_bias",
    "missing_control_buffer", "unbuffered_isolation_control", "buffer_on_switched_rail",
    "buffer_wrong_polarity", "buffer_output_capacitor", "weak_supervisor_pullup",
])
def test_analog_power_cycle_reset_requires_supervised_isolation_and_bounded_discharge(
    tmp_path, monkeypatch, defect
):
    state, board, names = _analog_power_cycle_artifact(defect=defect)
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(
        tmp_path, state, board,
        {"obligations": [{"check": {"kind": "gate", "gate": "external_reset_control_function"}}]},
    )
    assert (facts["gates"].get("external_reset_control_function") == "pass") is (defect is None)



def test_analog_power_cycle_reset_does_not_extrapolate_off_leakage_to_3v3(
    tmp_path, monkeypatch
):
    state, board, names = _analog_power_cycle_artifact(rail_voltage=3.3)
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(
        tmp_path, state, board,
        {"obligations": [{"check": {"kind": "gate", "gate": "external_reset_control_function"}}]},
    )
    assert facts["gates"].get("external_reset_control_function") != "pass"

def test_analog_power_cycle_reset_stays_unverified_without_an_authoritative_source_contract(
    tmp_path, monkeypatch
):
    state, board, names = _analog_power_cycle_artifact()
    state["architecture"]["requirements"] = []
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(
        tmp_path, state, board,
        {"obligations": [{"check": {"kind": "gate", "gate": "external_reset_control_function"}}]},
    )
    assert facts["gates"].get("external_reset_control_function") != "pass"


@pytest.mark.parametrize("identity", ["me6211c33m5g-n", "ap2112k-3.3trg1"])
@pytest.mark.parametrize(
    "fault",
    [None, "source_open", "source_unbound", "disabled", "missing_bypass",
     "insufficient_headroom", "overvoltage", "nc_connected", "self_powered"],
)
def test_linear_logic_supply_requires_powered_enabled_supported_circuit(identity, fault):
    record = reviewed_part(identity)
    connector = reviewed_part("b2b-xh-a(lf)(sn)")
    parts = [
        {"ref": "U1", "mpn": identity, "symbol": record.symbol, "footprint": record.footprint},
        {"ref": "J1", "mpn": connector.identity, "symbol": connector.symbol, "footprint": connector.footprint},
        {"ref": "C1", "symbol": "Device:C", "value": "1uF"},
        {"ref": "C2", "symbol": "Device:C", "value": "1uF"},
    ]
    pads = {
        ("U1", "1"): "SOURCE", ("U1", "2"): "RETURN",
        ("U1", "3"): "SOURCE", ("U1", "5"): "LOGIC",
        ("J1", "1"): "SOURCE", ("J1", "2"): "RETURN",
        ("C1", "1"): "SOURCE", ("C1", "2"): "RETURN",
        ("C2", "1"): "LOGIC", ("C2", "2"): "RETURN",
    }
    names = {
        ("U1", "1"): "VIN", ("U1", "2"): record.port_pins["ground"],
        ("U1", "3"): record.port_pins["enable"], ("U1", "4"): "NC",
        ("U1", "5"): "VOUT",
    }
    state = {"architecture": {
        "rail_voltages": {"SOURCE": 5.0, "LOGIC": 3.3, "RETURN": 0.0},
        "requirements": [{"role": "power_input", "ports": {"positive": "SOURCE", "ground": "RETURN"}}],
    }}
    if fault == "source_open":
        pads["J1", "1"] = "OPEN"
    elif fault == "source_unbound":
        state["architecture"]["requirements"] = []
    elif fault == "disabled":
        pads["U1", "3"] = "RETURN"
    elif fault == "missing_bypass":
        pads["C2", "1"] = "OPEN"
    elif fault == "insufficient_headroom":
        state["architecture"]["rail_voltages"]["SOURCE"] = 3.3
    elif fault == "overvoltage":
        state["architecture"]["rail_voltages"]["SOURCE"] = 7.4
    elif fault == "nc_connected":
        pads["U1", "4"] = "RETURN"
    elif fault == "self_powered":
        pads["U1", "1"] = pads["U1", "3"] = "LOGIC"
    observed = electrical_artifact_evidence._observed_linear_outputs(parts, names, pads, state, [])
    voltage = electrical_artifact_evidence._supply_voltage_observed(
        "LOGIC", "RETURN", parts, names, pads, state, observed,
    )
    if fault is None:
        assert voltage == 3.3
        assert observed[0]["current_a"] is None
    else:
        assert voltage is None


def test_programming_header_is_not_an_assumed_external_power_source():
    connector = reviewed_part("wj126v-5.0-02p-14-00a")
    header = {
        "ref": "J1", "mpn": connector.identity, "symbol": connector.symbol,
        "footprint": connector.footprint, "value": "UART PROGRAM",
        "recipe_role": "program_header",
    }
    pads = {("J1", "1"): "+3V3", ("J1", "2"): "GND"}
    state = {"architecture": {
        "rail_voltages": {"+3V3": 3.3, "GND": 0.0},
        "requirements": [{"role": "power_input", "ports": {
            "positive": "+3V3", "ground": "GND",
        }}],
    }}
    observed = electrical_artifact_evidence._supply_voltage_observed
    assert observed("+3V3", "GND", [header], {}, pads, state, []) is None
    actual_input = {**header, "value": "3V3 INPUT", "recipe_role": "power_input"}
    assert observed("+3V3", "GND", [actual_input], {}, pads, state, []) == 3.3


def test_r2r_proof_requires_complete_delivered_lowered_graph(tmp_path):
    from kicraft.design.models import Architecture, BOM, CircuitRequirement, Sheet
    from kicraft.server.stage_work_units import probe_architecture_construction

    requirement = CircuitRequirement.model_validate({
        "id": "r2r", "sheet": "DAC", "role": "analog_block", "family": "r2r-ladder",
        "parameters": {"bits": 8, "r_value": 10000, "two_r_value": 20000},
        "ports": {**{f"bit{i}": f"D{i}" for i in range(8)}, "output": "OUT", "gnd": "GND"},
        "obligations": [{"kind": "physical", "original_obligation_id": "r2r-network",
                         "component_class": "resistor-network"}],
    })
    architecture = Architecture(
        sheets=[Sheet(name="DAC", stem="DAC", function="R-2R DAC")],
        power_nets=[], inter_sheet_nets=[], requirements=[requirement],
    )
    compiled = probe_architecture_construction({
        "architecture": architecture.model_dump(mode="json"), "bom": {"parts": []},
    })
    bom = BOM.model_validate(compiled)
    state = {"architecture": architecture.model_dump(mode="json"), "bom": bom.model_dump(mode="json")}
    pads = {
        (endpoint.ref, str(endpoint.pin)): connection.net_name
        for connection in bom.connections for endpoint in connection.endpoints
    }
    def board(graph):
        return _Board([
            _Footprint(part.ref, part.footprint, [
                (pin, net) for (ref, pin), net in graph.items() if ref == part.ref
            ])
            for part in bom.parts
        ])

    intact = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board(pads), {})
    assert intact["net_paths"].get("r2r_ladder") is True
    broken = dict(pads)
    broken[("R1", "2")] = "BROKEN_LADDER"
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board(broken), {})
    assert facts["net_paths"].get("r2r_ladder") is not True
