from __future__ import annotations

import pytest

from kicraft.design.actuator_circuits import _a4988_socket, _tb6612, _uln2003
from kicraft.design.models import CircuitRequirement
from kicraft.eval.actuator_artifact_evidence import actuator_circuit_facts


def _requirement(family: str, ports: dict[str, str], *, channels: int, exact: str | None = None) -> CircuitRequirement:
    return CircuitRequirement(id="actuator", sheet="ACTUATOR", role="driver", family=family, exact_part=exact, parameters={"channel_count": channels}, ports=ports)


def _uln_ports(channels: int) -> dict[str, str]:
    ports = {"vmotor": "VMOTOR", "gnd": "GND"}
    for number in range(1, channels + 1):
        ports.update({f"control_{number}": f"CTRL{number}", f"load_{number}_positive": "VMOTOR", f"load_{number}_negative": f"OUT{number}"})
    return ports


def _tb_ports(channels: int) -> dict[str, str]:
    ports = {"vcc": "3V3", "vmotor": "VMOTOR", "gnd": "GND", "stby": "TB_STBY"}
    for number in range(1, channels + 1):
        ports.update({f"motor_{number}_in1": f"M{number}_IN1", f"motor_{number}_in2": f"M{number}_IN2", f"motor_{number}_pwm": f"M{number}_PWM", f"motor_{number}_positive": f"M{number}P", f"motor_{number}_negative": f"M{number}N"})
    return ports


def test_uln2003_scales_eight_physical_outputs_and_unused_inputs_are_tied_low():
    artifact = _uln2003(_requirement("uln2003", _uln_ports(8), channels=8), relay=False)
    assert artifact is not None
    assert sum(group.quantity for group in artifact.groups if group.role == "driver") == 2
    assert len({pin.net for pin in artifact.pins if pin.role == "driver" and pin.pin in {"10", "11", "12", "13", "14", "15", "16"}}) == 8
    assert all(pin.net == "GND" for pin in artifact.pins if pin.role == "driver" and pin.pin in {"2", "3", "4", "5", "6", "7"} and pin.index == 1)


def test_uln2003_rejects_merged_load_outputs_and_explicit_rail_conflicts():
    merged = _uln_ports(2)
    merged["load_2_negative"] = merged["load_1_negative"]
    assert _uln2003(_requirement("uln2003", merged, channels=2), relay=False) is None
    conflict = _uln_ports(2)
    conflict["vmotor"] = "GND"
    conflict["load_1_positive"] = "GND"
    conflict["load_2_positive"] = "GND"
    assert _uln2003(_requirement("uln2003", conflict, channels=2), relay=False) is None


def test_relay_outputs_use_real_coils_flyback_driver_and_contact_terminals():
    ports = {"vmotor": "VMOTOR", "gnd": "GND"}
    for number in range(1, 4):
        ports.update({
            f"control_{number}": f"CTRL{number}",
            f"relay_{number}_common": f"R{number}_COM",
            f"relay_{number}_normally_open": f"R{number}_NO",
            f"relay_{number}_normally_closed": f"R{number}_NC",
        })
    artifact = _uln2003(_requirement("relay-output", ports, channels=3), relay=True)
    assert artifact is not None
    assert sum(group.quantity for group in artifact.groups if group.role == "relay") == 3
    assert sum(group.quantity for group in artifact.groups if group.role == "relay_contact_terminal") == 3
    assert {pin.net for pin in artifact.pins if pin.role == "driver" and pin.pin == "9"} == {"VMOTOR"}
    coils = {
        pin.index: pin.net for pin in artifact.pins
        if pin.role == "relay" and pin.pin == "4"
    }
    assert len(set(coils.values())) == 3
    assert not set(coils.values()) & set(ports.values())
    assert {
        pin.net for pin in artifact.pins
        if pin.role == "driver" and pin.pin in {"16", "15", "14"}
    } == set(coils.values())
    assert not any(group.role == "load_terminal" for group in artifact.groups)

def test_relay_voltage_selects_coils_and_correctly_oriented_tvs():
    ports = {"vmotor": "VMOTOR", "gnd": "GND"}
    for number in (1, 2):
        ports.update({
            f"control_{number}": f"CTRL{number}",
            f"relay_{number}_common": f"R{number}_COM",
            f"relay_{number}_normally_open": f"R{number}_NO",
            f"relay_{number}_normally_closed": f"R{number}_NC",
        })
    base = _requirement("relay-output", ports, channels=2)
    for voltage, relay_mpn, tvs_mpn in ((5.0, "SRD-05VDC-SL-C", "SMBJ5.0A-13-F"), (24.0, "SRD-24VDC-SL-C", "SMBJ26A")):
        requirement = base.model_copy(update={"parameters": {"channel_count": 2, "motor_voltage_v": voltage}})
        artifact = _uln2003(requirement, relay=True)
        assert artifact is not None
        assert next(group.mpn for group in artifact.groups if group.role == "relay") == relay_mpn
        assert next(group.mpn for group in artifact.groups if group.role == "motor_tvs") == tvs_mpn
        assert {pin.pin: pin.net for pin in artifact.pins if pin.role == "motor_tvs"} == {"1": "VMOTOR", "2": "GND"}
        assert {pin.net for pin in artifact.pins if pin.role == "driver" and pin.pin == "9"} == {"VMOTOR"}
    unsupported = base.model_copy(update={"parameters": {"channel_count": 2, "motor_voltage_v": 12.0}})
    assert _uln2003(unsupported, relay=True) is None


def test_tb6612_scales_every_channel_with_all_supply_support_and_safe_controls():
    artifact = _tb6612(_requirement("tb6612fng", _tb_ports(8), channels=8, exact="TB6612FNG"))
    assert artifact is not None
    assert sum(group.quantity for group in artifact.groups if group.role == "driver") == 4
    assert sum(group.quantity for group in artifact.groups if group.role == "motor_terminal") == 8
    assert sum(group.quantity for group in artifact.groups if group.role == "control_pulldown") == 24
    assert {pin.pin for pin in artifact.pins if pin.role == "driver"} >= {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"}
    assert {pin.pin: pin.net for pin in artifact.pins if pin.role == "motor_tvs"} == {"1": "VMOTOR", "2": "GND"}


@pytest.mark.parametrize("identity,lcsc,voltage", [
    ("SMBJ5.0A-13-F", "C110528", 5.0),
    ("SMBJ26A", "C726831", 24.0),
])
def test_tvs_qualification_requires_exact_supplier_and_polarity(identity, lcsc, voltage):
    from kicraft.design.part_identity import reviewed_part
    from kicraft.eval.actuator_artifact_evidence import _tvs_clamps

    record = reviewed_part(identity)
    part = {"ref": "D1", "mpn": identity, "symbol": record.symbol,
            "footprint": record.footprint, "sourcing_note": f"LCSC {lcsc}"}
    pads = {("D1", "1"): "VM", ("D1", "2"): "GND"}
    assert _tvs_clamps([part], pads, "VM", {"GND"}, voltage)
    assert not _tvs_clamps([part], {("D1", "1"): "GND", ("D1", "2"): "VM"}, "VM", {"GND"}, voltage)
    assert not _tvs_clamps([{**part, "sourcing_note": "LCSC C113974"}], pads, "VM", {"GND"}, voltage)
    assert not _tvs_clamps([{**part, "symbol": "Device:D", "footprint": "Diode_SMD:D_SMB"}], pads, "VM", {"GND"}, voltage)
    assert not _tvs_clamps([{**part, "mpn": "OTHER-MANUFACTURER"}], pads, "VM", {"GND"}, voltage)

@pytest.mark.parametrize("voltage,identity,lcsc", [
    (5.0, "SMBJ5.0A-13-F", "C110528"), (24.0, "SMBJ26A", "C726831"),
])
def test_relay_tvs_source_survives_fab_export(tmp_path, voltage, identity, lcsc):
    import csv
    from kicraft.design.part_identity import reviewed_part
    from kicraft.design.synthesis.fab_export import _write_bom_csv

    ports = {"vmotor": "VMOTOR", "gnd": "GND"}
    for number in (1, 2):
        ports.update({f"control_{number}": f"CTRL{number}",
                      f"relay_{number}_common": f"C{number}",
                      f"relay_{number}_normally_open": f"NO{number}",
                      f"relay_{number}_normally_closed": f"NC{number}"})
    requirement = _requirement("relay-output", ports, channels=2).model_copy(
        update={"parameters": {"channel_count": 2, "motor_voltage_v": voltage}})
    group = next(g for g in _uln2003(requirement, relay=True).groups if g.role == "motor_tvs")
    record = reviewed_part(identity)
    assert (group.mpn, group.symbol, group.footprint) == (identity, record.symbol, record.footprint)
    path = tmp_path / "bom.csv"
    _write_bom_csv(path, [{**group.model_dump(), "ref": "D1"}])
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert (row["mpn"], row["lcsc"]) == (identity, lcsc)



def test_tb6612_rejects_merged_control_and_rail_aliases():
    ports = _tb_ports(2)
    ports["motor_1_pwm"] = ports["motor_1_in1"]
    assert _tb6612(_requirement("tb6612fng", ports, channels=2, exact="TB6612FNG")) is None
    ports = _tb_ports(2)
    ports["vmotor"] = ports["vcc"]
    assert _tb6612(_requirement("tb6612fng", ports, channels=2, exact="TB6612FNG")) is None


def test_a4988_socket_is_carrier_mating_footprint_not_bare_ic():
    ports = {"vmotor": "VMOTOR", "vdd": "3V3", "gnd": "GND", "enable_n": "ENABLE_N", "ms1": "MS1", "ms2": "MS2", "ms3": "MS3", "reset_n": "RESET_N", "sleep_n": "SLEEP_N", "step": "STEP", "dir": "DIR", "coil_1a": "1A", "coil_1b": "1B", "coil_2a": "2A", "coil_2b": "2B"}
    artifact = _a4988_socket(_requirement("a4988-driver-socket", ports, channels=2, exact="A4988 driver socket"))
    assert artifact is not None
    socket = next(group for group in artifact.groups if group.role == "carrier_socket")
    assert socket.symbol == "a4988-carrier-socket:A4988_CARRIER_SOCKET"
    assert socket.footprint == "a4988-carrier-socket:A4988_CARRIER_SOCKET_POLolu_1182"
    assert not any(group.value == "A4988SETTR-T" for group in artifact.groups)


def test_uln_observer_rejects_wrong_com_missing_pull_and_merged_output():
    parts = [
        {"ref": "U1", "mpn": "ULN2003ADR", "symbol": "uln2003adr:ULN2003ADR", "footprint": "uln2003adr:SOIC-16_L9.9-W3.9-P1.27-LS6.0-BL"},
        {"ref": "J1", "mpn": "WJ126V-5.0-02P-14-00A", "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P", "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P"},
        {"ref": "J2", "mpn": "WJ126V-5.0-02P-14-00A", "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P", "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P"},
    ]
    names = {("U1", "1"): "IN1", ("U1", "2"): "IN2", ("U1", "8"): "E", ("U1", "9"): "COM", ("U1", "16"): "OUT1", ("U1", "15"): "OUT2"}
    pads = {("U1", "1"): "CTRL1", ("U1", "2"): "CTRL2", ("U1", "8"): "GND", ("U1", "9"): "GND", ("U1", "16"): "OUT", ("U1", "15"): "OUT", ("J1", "1"): "VMOTOR", ("J1", "2"): "OUT", ("J2", "1"): "VMOTOR", ("J2", "2"): "OUT"}
    facts = actuator_circuit_facts(parts, names, pads)
    assert "uln2003_inductive_load_path" not in facts["net_paths"]
    assert "uln2003_inductive_clamp" not in facts["net_paths"]
    assert "uln2003_safe_default" not in facts["net_paths"]


def test_a4988_observer_rejects_bare_ic_representation():
    facts = actuator_circuit_facts([{"ref": "U1", "mpn": "A4988SETTR-T", "symbol": "a4988:A4988SETTR-T", "footprint": "a4988:WQFN-28_L5.0-W5.0-P0.50-BL-EP3.2"}], {}, {})
    assert "a4988_carrier_socket_representation" not in facts["net_paths"]
    assert not facts["gates"]

def test_tb_observer_rejects_missing_stby_support_clamp_and_motor_contact():
    parts = [{"ref": "U1", "mpn": "TB6612FNG", "symbol": "tb6612fng:TB6612FNG", "footprint": "tb6612fng:SSOP24-P-300-0.65A"}]
    names = {
        ("U1", "1"): "AO1", ("U1", "2"): "AO1", ("U1", "5"): "AO2", ("U1", "6"): "AO2",
        ("U1", "7"): "BO2", ("U1", "8"): "BO2", ("U1", "11"): "BO1", ("U1", "12"): "BO1",
        ("U1", "3"): "PGND1", ("U1", "4"): "PGND1", ("U1", "9"): "PGND2", ("U1", "10"): "PGND2",
        ("U1", "13"): "VM2", ("U1", "14"): "VM3", ("U1", "24"): "VM1", ("U1", "18"): "GND",
        ("U1", "19"): "STBY", ("U1", "20"): "VCC", ("U1", "15"): "PWMB", ("U1", "16"): "BIN2",
        ("U1", "17"): "BIN1", ("U1", "21"): "AIN1", ("U1", "22"): "AIN2", ("U1", "23"): "PWMA",
    }
    pads = {
        endpoint: "GND" if name in {"GND", "PGND1", "PGND2"} else "VMOTOR"
        if name.startswith("VM") else name
        for endpoint, name in names.items()
    }
    facts = actuator_circuit_facts(parts, names, pads)
    assert "tb6612_motor_channel_path" not in facts["net_paths"]
    assert "tb6612_control_and_safe_default" not in facts["net_paths"]
    assert "tb6612_support_and_clamp" not in facts["net_paths"]
    assert "external_motor_controller_function" not in facts["gates"]
    assert "external_motor_screw_terminal_binding" not in facts["gates"]


def _a4988_socket_facts(
    *, unsafe_default: bool = False, motor_terminal: bool = True,
    jst_motor_endpoints: bool = False,
):
    socket = {
        "ref": "J1",
        "symbol": "a4988-carrier-socket:A4988_CARRIER_SOCKET",
        "footprint": "a4988-carrier-socket:A4988_CARRIER_SOCKET_POLolu_1182",
    }
    terminal = {
        "ref": "J2", "mpn": "WJ126V-5.0-04P-14-00A",
        "symbol": "screw-terminal-5mm-4p:WJ126V-5.0-4P",
        "footprint": "screw-terminal-5mm-4p:CONN-TH_4P-P5.00_WJ126V-5.0-4P-1",
    }
    controls = ["ENABLE", "RESET", "SLEEP", "STEP", "DIR"]
    parts = [socket, *[
        {
            "ref": f"JCTRL{index}", "mpn": "WJ126V-5.0-02P-14-00A",
            "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P",
            "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
        }
        for index, _ in enumerate(controls, 1)
    ]]
    if motor_terminal:
        parts.append(terminal)
    if jst_motor_endpoints:
        parts += [
            {
                "ref": "JX1", "mpn": "B2B-XH-A(LF)(SN)",
                "symbol": "b2b-xh-a-lf-sn:B2B-XH-A",
                "footprint": "b2b-xh-a-lf-sn:CONN-TH_B2B-XH-A-LF-SN",
            },
            {
                "ref": "JX2", "mpn": "B2B-XH-A(LF)(SN)",
                "symbol": "b2b-xh-a-lf-sn:B2B-XH-A",
                "footprint": "b2b-xh-a-lf-sn:CONN-TH_B2B-XH-A-LF-SN",
            },
        ]
    parts += [
        {
            "ref": "JPWR1", "mpn": "WJ126V-5.0-02P-14-00A",
            "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P",
            "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
        },
        {
            "ref": "JPWR2", "mpn": "WJ126V-5.0-02P-14-00A",
            "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P",
            "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
        },
    ]
    parts += [
        {"ref": "REN", "symbol": "Device:R"},
        *[
            {"ref": f"R{signal}", "symbol": "Device:R"}
            for signal in ("MS1", "MS2", "MS3", "RESET", "SLEEP")
            if not (unsafe_default and signal == "SLEEP")
        ],
        {"ref": "C1", "symbol": "Device:C", "value": "100uF"},
        {"ref": "C2", "symbol": "Device:C", "value": "100nF"},
    ]
    nets = {
        "1": "ENABLE", "2": "MS1", "3": "MS2", "4": "MS3",
        "5": "RESET", "6": "SLEEP", "7": "STEP", "8": "DIR",
        "9": "GND", "10": "VDD", "11": "COIL1B", "12": "COIL1A",
        "13": "COIL2A", "14": "COIL2B", "15": "GND", "16": "VMOT",
    }
    pads = {("J1", pin): net for pin, net in nets.items()}
    if motor_terminal:
        pads.update({
            ("J2", "1"): "COIL1A", ("J2", "2"): "COIL1B",
            ("J2", "3"): "COIL2A", ("J2", "4"): "COIL2B",
        })
    if jst_motor_endpoints:
        pads.update({
            ("JX1", "1"): "COIL1A", ("JX1", "2"): "COIL1B",
            ("JX2", "1"): "COIL2A", ("JX2", "2"): "COIL2B",
        })
    for index, signal in enumerate(controls, 1):
        pads[("JCTRL" + str(index), "1")] = signal
        pads[("JCTRL" + str(index), "2")] = "AUX" + str(index)
    pads.update({
        ("JPWR1", "1"): "VDD", ("JPWR1", "2"): "GND",
        ("JPWR2", "1"): "VMOT", ("JPWR2", "2"): "GND",
    })
    pads.update({
        ("REN", "1"): "VDD", ("REN", "2"): "ENABLE",
        ("C1", "1"): "VMOT", ("C1", "2"): "GND",
        ("C2", "1"): "VMOT", ("C2", "2"): "GND",
    })
    for signal in ("MS1", "MS2", "MS3", "RESET", "SLEEP"):
        pads[("R" + signal, "1")] = signal
        pads[("R" + signal, "2")] = "GND"
    return actuator_circuit_facts(parts, {}, pads)


def test_a4988_function_gates_require_carrier_controls_safe_state_and_motor_terminal():
    facts = _a4988_socket_facts()
    assert facts["gates"] == {
        "a4988_driver_socket": "pass",
        "a4988_socket_motor_terminal_binding": "pass",
    }

    assert not _a4988_socket_facts(unsafe_default=True)["gates"]
    assert not _a4988_socket_facts(motor_terminal=False)["gates"]
    generic = _a4988_socket_facts(
        motor_terminal=False, jst_motor_endpoints=True,
    )
    assert generic["gates"] == {
        "a4988_driver_socket": "pass",
    }


def _relay_facts(*, broken_control: bool = False, missing_return: bool = False,
                 mcu_controls: bool = False, powered_mcu_refs: set[str] | None = None):
    parts = [
        {
            "ref": "U1", "mpn": "ULN2003ADR",
            "symbol": "uln2003adr:ULN2003ADR",
            "footprint": "uln2003adr:SOIC-16_L9.9-W3.9-P1.27-LS6.0-BL",
        },
        *[
            {
                "ref": f"K{index}", "mpn": "SRD-05VDC-SL-C",
                "symbol": "srd-05vdc-sl-c:SRD-05VDC-SL-C",
                "footprint": "srd-05vdc-sl-c:RELAY-TH_SRD-XXVDC-XL-C",
            }
            for index in (1, 2)
        ],
        *[
            {
                "ref": f"JOUT{index}", "mpn": "WJ126V-5.0-03P-14-00A",
                "symbol": "screw-terminal-5mm-3p:WJ126V-5.0-3P",
                "footprint": "screw-terminal-5mm-3p:CONN-TH_3P-P5.00_WJ126V-5.0-3P",
            }
            for index in (1, 2)
        ],
        *[
            {
                "ref": f"JCTRL{index}", "mpn": "WJ126V-5.0-02P-14-00A",
                "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P",
                "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
            }
            for index in (1, 2)
            if not mcu_controls and not (broken_control and index == 2)
        ],
        {"ref": "R1", "symbol": "Device:R"},
        {"ref": "R2", "symbol": "Device:R"},
        {"ref": "D1", "mpn": "SMBJ5.0A-13-F", "symbol": "diodes-smbj5-0a:SMBJ5.0A-13-F", "footprint": "diodes-smbj5-0a:SMB_L4.6-W3.6-LS5.3-RD", "sourcing_note": "LCSC C110528"},
    ]
    if mcu_controls:
        parts.append({
            "ref": "U2", "mpn": "ESP32-C3-MINI-1-H4X",
            "symbol": "esp32-c3-mini-1-h4x:ESP32-C3-MINI-1-H4X",
            "footprint": "esp32-c3-mini-1-h4x:WIFIM-SMD_ESP32-C3-MINI-1",
        })
    names = {
        ("U1", "1"): "IN1", ("U1", "2"): "IN2", ("U1", "8"): "E",
        ("U1", "9"): "COM", ("U1", "16"): "OUT1", ("U1", "15"): "OUT2",
    }
    pads = {
        ("U1", "1"): "CTRL1", ("U1", "2"): "CTRL2",
        ("U1", "8"): "FLOAT" if missing_return else "GND",
        ("U1", "9"): "VM", ("U1", "16"): "COIL1", ("U1", "15"): "COIL2",
        ("R1", "1"): "CTRL1", ("R1", "2"): "GND",
        ("R2", "1"): "CTRL2", ("R2", "2"): "GND",
        ("D1", "1"): "VM", ("D1", "2"): "GND",
    }
    for index in (1, 2):
        pads.update({
            (f"K{index}", "1"): "VM", (f"K{index}", "4"): f"COIL{index}",
            (f"K{index}", "5"): f"COM{index}", (f"K{index}", "2"): f"NO{index}",
            (f"K{index}", "3"): f"NC{index}",
            (f"JOUT{index}", "1"): f"COM{index}",
            (f"JOUT{index}", "2"): f"NO{index}",
            (f"JOUT{index}", "3"): f"NC{index}",
        })
        if mcu_controls:
            pads[("U2", str(17 + index))] = f"CTRL{index}"
        elif not (broken_control and index == 2):
            pads[(f"JCTRL{index}", "1")] = f"CTRL{index}"
            pads[(f"JCTRL{index}", "2")] = f"AUX{index}"
    return actuator_circuit_facts(
        parts, names, pads,
        observed_outputs=[{"output_net": "VM", "ground_net": "GND", "voltage": 5.0}],
        powered_mcu_refs=powered_mcu_refs,
    )


def test_relay_gates_require_complete_coil_control_return_and_contact_terminals():
    facts = _relay_facts()
    assert facts["gates"]["external_relay_control_function"] == "pass"
    assert facts["gates"]["external_relay_outputs_on_screw_terminals"] == "pass"
    assert facts["gates"]["external_actuator_driver_function"] == "pass"

    for facts in (_relay_facts(broken_control=True), _relay_facts(missing_return=True)):
        assert "external_relay_control_function" not in facts["gates"]
        assert "external_relay_outputs_on_screw_terminals" not in facts["gates"]


def test_relay_mcu_controls_require_physically_powered_complete_mcu():
    powered = _relay_facts(mcu_controls=True, powered_mcu_refs={"U2"})
    assert powered["gates"]["external_relay_control_function"] == "pass"
    assert powered["gates"]["external_relay_outputs_on_screw_terminals"] == "pass"
    unpowered = _relay_facts(mcu_controls=True, powered_mcu_refs=set())
    assert "external_relay_control_function" not in unpowered["gates"]
    assert "external_relay_outputs_on_screw_terminals" not in unpowered["gates"]


def test_tb6612_function_uses_its_actual_return_not_another_ground_named_net():
    from kicraft.design.part_identity import reviewed_part
    from kicraft.design.synthesis.symbol_pinout import lookup_pins
    from kicraft.server.stage_work_units import probe_architecture_construction

    requirement = _requirement("tb6612fng", _tb_ports(2), channels=2, exact="TB6612FNG")
    constructed = probe_architecture_construction({"architecture": {
        "topologies": {}, "rail_voltages": {"3V3": 3.3, "VMOTOR": 5.0},
        "comms_protocols": [], "mcu_present": False,
        "sheets": [{"name": "ACTUATOR", "stem": "ACTUATOR", "function": "Motor driver"}],
        "power_nets": ["3V3", "VMOTOR", "GND"], "inter_sheet_nets": [],
        "requirements": [requirement.model_dump(mode="json")],
    }})
    assert constructed is not None
    parts = constructed["parts"]
    base_pads = {
        (endpoint["ref"], endpoint["pin"]): connection["net_name"]
        for connection in constructed["connections"] for endpoint in connection["endpoints"]
    }
    names = {
        (part["ref"], str(pin["number"])): str(pin.get("name") or "")
        for part in parts for pin in lookup_pins(part["symbol"], all_units=True)["pins"]
    }
    driver = next(part["ref"] for part in parts if part.get("mpn") == "TB6612FNG")
    connector = reviewed_part("WJ126V-5.0-02P-14-00A")
    for index, net in enumerate(["TB_STBY", *[
        f"M{channel}_{role}" for channel in (1, 2) for role in ("IN1", "IN2", "PWM")
    ]]):
        ref = f"J{900 + index}"
        parts.append({"ref": ref, "mpn": connector.identity, "symbol": connector.symbol, "footprint": connector.footprint})
        base_pads[ref, "1"], base_pads[ref, "2"] = net, "GND"
    for ground in ("GND", "RETURN"):
        for broken in ((), ("4",), ("3", "4", "9", "10", "18")):
            pads = {endpoint: ground if net == "GND" else net for endpoint, net in base_pads.items()}
            for pin in broken:
                pads[driver, pin] = "FLOAT"
            facts = actuator_circuit_facts(parts, names, pads, observed_outputs=[
                {"output_net": "VMOTOR", "ground_net": ground, "voltage": 5.0},
                {"output_net": "3V3", "ground_net": ground, "voltage": 3.3},
            ])
            assert (facts["gates"].get("external_motor_controller_function") == "pass") is (not broken)


def _compiled_relay_case(*, physical_count: int | None = None):
    from kicraft.design.synthesis.symbol_pinout import lookup_pins
    from kicraft.server.stage_work_units import probe_architecture_construction

    ports = {"vmotor": "VMOTOR", "gnd": "GND"}
    for number in (1, 2):
        ports.update({
            f"control_{number}": f"CTRL{number}",
            f"relay_{number}_common": f"R{number}_COM",
            f"relay_{number}_normally_open": f"R{number}_NO",
            f"relay_{number}_normally_closed": f"R{number}_NC",
        })
    relay = _requirement("relay-output", ports, channels=2, exact="ULN2003ADR")
    if physical_count is not None:
        relay = CircuitRequirement.model_validate({
            **relay.model_dump(mode="json"),
            "obligations": [
                {"kind": "physical", "component_class": "relay", "original_obligation_id": "relay"},
                {"kind": "quantity", "subject": "relays", "minimum": physical_count,
                 "original_obligation_id": "relay_count"},
            ],
        })
    interface = CircuitRequirement(
        id="host", sheet="ACTUATOR", role="connector", family="pin-header",
        ports={"pin1": "VMOTOR", "pin2": "GND", "pin3": "CTRL1", "pin4": "CTRL2"},
    )
    architecture = {
        "topologies": {}, "rail_voltages": {"VMOTOR": 5.0, "GND": 0.0},
        "comms_protocols": [], "mcu_present": False,
        "sheets": [{"name": "ACTUATOR", "stem": "ACTUATOR", "function": "Relay bank"}],
        "power_nets": ["VMOTOR", "GND"], "inter_sheet_nets": [],
        "requirements": [relay.model_dump(mode="json"), interface.model_dump(mode="json")],
    }
    constructed = probe_architecture_construction({"architecture": architecture})
    assert constructed is not None
    names = {
        (part["ref"], str(pin["number"])): str(pin.get("name") or "")
        for part in constructed["parts"]
        for pin in lookup_pins(part["symbol"], all_units=True)["pins"]
    }
    return architecture, constructed, names


def test_compiled_relay_public_interface_proves_coils_controls_clamp_and_contacts():
    architecture, constructed, names = _compiled_relay_case()
    parts = constructed["parts"]
    pads = {
        (endpoint["ref"], endpoint["pin"]): connection["net_name"]
        for connection in constructed["connections"] for endpoint in connection["endpoints"]
    }
    first_relay = next(part["ref"] for part in parts if part.get("mpn") == "SRD-05VDC-SL-C")
    driver = next(part["ref"] for part in parts if part.get("mpn") == "ULN2003ADR")
    host = next(part["ref"] for part in parts if part.get("lowering_requirement_id") == "host")
    observed = [{"output_net": "VMOTOR", "ground_net": "GND", "voltage": 5.0}]
    intact = actuator_circuit_facts(parts, names, pads, observed_outputs=observed)
    assert intact["gates"]["external_relay_control_function"] == "pass"
    assert intact["channel_counts"]["relay_control_channel"] == 2
    for endpoint in ((first_relay, "4"), (host, "3"), (driver, "9"), (first_relay, "5")):
        broken = dict(pads)
        broken.pop(endpoint)
        facts = actuator_circuit_facts(parts, names, broken, observed_outputs=observed)
        assert "external_relay_control_function" not in facts["gates"]


def test_relay_coil_nodes_cannot_be_externally_claimed_or_contact_aliased():
    from kicraft.design.lowering import lower_requirement
    import pytest

    architecture, _, _ = _compiled_relay_case()
    requirement = CircuitRequirement.model_validate(architecture["requirements"][0])
    externalized = requirement.model_copy(update={
        "ports": {**requirement.ports, "load_1_positive": "VMOTOR", "load_1_negative": "COIL"},
    })
    with pytest.raises(ValueError):
        lower_requirement(externalized)
    collision = requirement.model_copy(update={
        "ports": {
            **requirement.ports,
            "relay_1_common": "__KICRAFT_RELAY__actuator__COIL_1",
        },
    })
    with pytest.raises(ValueError):
        lower_requirement(collision)


def test_composite_relay_physical_demand_counts_the_two_real_support_relays():
    from kicraft.design.models import Architecture, BOM
    from kicraft.design.synthesis.validation import check_requirement_physical_realization

    architecture, constructed, _ = _compiled_relay_case(physical_count=2)
    bom = BOM.model_validate(constructed)
    assert len({part.ref for part in bom.parts if part.mpn == "SRD-05VDC-SL-C"}) == 2
    result = check_requirement_physical_realization(Architecture.model_validate(architecture), bom)
    assert result.ok, result.offenders


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_relay", "wrong_coil_identity", "wrong_owner", "model_owned",
        "open_coil", "missing_principal", "borrowed_bank", "insufficient_count",
    ],
)
def test_composite_relay_physical_demand_rejects_unproven_support(corruption):
    from kicraft.design.models import Architecture, BOM
    from kicraft.design.synthesis.validation import check_requirement_physical_realization

    payload, constructed, _ = _compiled_relay_case(physical_count=2)
    architecture = Architecture.model_validate(payload)
    bom = BOM.model_validate(constructed)
    relay = next(part for part in bom.parts if part.mpn == "SRD-05VDC-SL-C")
    driver = next(part for part in bom.parts if part.mpn == "ULN2003ADR")
    if corruption == "missing_relay":
        bom.parts = [part for part in bom.parts if part.ref != relay.ref]
    elif corruption == "wrong_coil_identity":
        relay.mpn = "SRD-12VDC-SL-C"
    elif corruption == "wrong_owner":
        relay.lowering_requirement_id = "another_requirement"
    elif corruption == "model_owned":
        relay.resolution_source = "llm"
    elif corruption == "open_coil":
        for connection in bom.connections:
            connection.endpoints = [
                endpoint for endpoint in connection.endpoints
                if (endpoint.ref, endpoint.pin) != (relay.ref, "4")
            ]
    elif corruption == "missing_principal":
        bom.parts = [part for part in bom.parts if part.ref != driver.ref]
    elif corruption == "borrowed_bank":
        architecture.requirements.append(
            architecture.requirements[0].model_copy(update={"id": "second_bank"}, deep=True)
        )
    elif corruption == "insufficient_count":
        architecture.requirements[0].obligations[1].minimum = 3
    result = check_requirement_physical_realization(architecture, bom)
    assert not result.ok, corruption


def test_final_bom_realizes_composite_relay_count_before_wiring_stage():
    from kicraft.design.models import Architecture, BOM
    from kicraft.design.synthesis.validation import check_requirement_physical_realization
    from kicraft.server.stage_work_units import (
        aggregate_stage_work_units,
        deterministic_bom_candidate,
        plan_stage_work_units,
        validate_unit_candidate,
    )

    architecture, _, _ = _compiled_relay_case(physical_count=2)
    state = {"architecture": architecture}
    units = plan_stage_work_units("bom", state, {})
    candidates = {}
    for unit in units:
        candidate = deterministic_bom_candidate(unit, state)
        assert candidate is not None
        candidates[unit.unit_id] = validate_unit_candidate(
            unit, candidate, {**state, "_trusted_deterministic_candidate": True}, {}
        )
    payload, *_ = aggregate_stage_work_units(
        "bom", units, candidates, state,
        unit_sources={unit.unit_id: "lowerer" for unit in units},
    )
    result = check_requirement_physical_realization(
        Architecture.model_validate(architecture), BOM.model_validate(payload)
    )
    assert result.ok, result.offenders


@pytest.mark.parametrize("rail", ["VMOTOR", "3V3", "GND"])
def test_a4988_socket_rejects_coil_output_short_to_supply(rail):
    ports = {
        "vmotor": "VMOTOR", "vdd": "3V3", "gnd": "GND", "enable_n": "ENABLE_N",
        "ms1": "MS1", "ms2": "MS2", "ms3": "MS3", "reset_n": "RESET_N",
        "sleep_n": "SLEEP_N", "step": "STEP", "dir": "DIR",
        "coil_1a": rail, "coil_1b": "1B", "coil_2a": "2A", "coil_2b": "2B",
    }
    assert _a4988_socket(_requirement("a4988-driver-socket", ports, channels=2)) is None
