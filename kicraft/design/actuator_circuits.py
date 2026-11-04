"""Source-backed motor, relay, and carrier-socket circuit lowerers.

These lowerers intentionally publish the physical board-side interfaces.  They do
not claim motor thermal, braking-energy, or assembled-load qualification.
"""
from __future__ import annotations

import math
from typing import Iterable

from kicraft.design.models import CircuitRequirement
from kicraft.design.lowering import (
    LoweringArtifact,
    LoweringCalculation,
    LoweringGroup,
    LoweringNoConnect,
    LoweringPin,
    RegisteredLowerer,
    _artifact,
    register_lowerer,
)

_TI_ULN = "https://www.ti.com/lit/ds/symlink/uln2003a.pdf"
_TOSHIBA_TB = "https://toshiba.semicon-storage.com/info/TB6612FNG_datasheet_en_20141001.pdf?did=10660&prodName=TB6612FNG"
_POLOLU_A4988 = "https://www.pololu.com/product-info-merged/1182"
_SONGLE_SRD = "https://datasheet.lcsc.com/datasheet/pdf/9ddfe54b7529126320ff1ddcabf6bab7.pdf?productCode=C15840"
_SMBJ26 = "https://datasheet.lcsc.com/datasheet/pdf/a5de6b41e5c0fc8fd537b2d1bace2e83.pdf?productCode=C726831"


def _group(role: str, prefix: str, value: str, symbol: str, footprint: str, *, quantity: int = 1, mpn: str | None = None, datasheet: str | None = None, sourcing_note: str | None = None) -> LoweringGroup:
    return LoweringGroup(role=role, reference_prefix=prefix, value=value, symbol=symbol, footprint=footprint, quantity=quantity, mpn=mpn, datasheet=datasheet, sourcing_note=sourcing_note)


def _resistor(role: str, value: str, quantity: int) -> LoweringGroup:
    return _group(role, "R", value, "Device:R", "Resistor_SMD:R_0603_1608Metric", quantity=quantity)


def _capacitor(role: str, value: str, quantity: int) -> LoweringGroup:
    return _group(role, "C", value, "Device:C", "Capacitor_SMD:C_0805_2012Metric", quantity=quantity)


def _terminal(role: str, quantity: int) -> LoweringGroup:
    return _group(role, "J", "WJ126V-5.0-2P", "screw-terminal-5mm-2p:WJ126V-5.0-2P", "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P", quantity=quantity, mpn="WJ126V-5.0-02P-14-00A")


def _relay_terminal(role: str, quantity: int) -> LoweringGroup:
    return _group(role, "J", "WJ126V-5.0-3P", "screw-terminal-5mm-3p:WJ126V-5.0-3P", "screw-terminal-5mm-3p:CONN-TH_3P-P5.00_WJ126V-5.0-3P", quantity=quantity, mpn="WJ126V-5.0-03P-14-00A")


def _tvs() -> LoweringGroup:
    return _group("motor_tvs", "D", "SMBJ5.0A-13-F", "diodes-smbj5-0a:SMBJ5.0A-13-F", "diodes-smbj5-0a:SMB_L4.6-W3.6-LS5.3-RD", mpn="SMBJ5.0A-13-F", datasheet="https://www.diodes.com/assets/Datasheets/ds19002.pdf", sourcing_note="LCSC C110528")


def _required_ports(requirement: CircuitRequirement, keys: Iterable[str]) -> dict[str, str] | None:
    required = set(keys)
    if set(requirement.ports) != required or any(not str(value).strip() for value in requirement.ports.values()):
        return None
    return {key: str(value) for key, value in requirement.ports.items()}


def _different(*nets: str) -> bool:
    return len(set(nets)) == len(nets)


def _channels(parameters: dict, allowed: set[int]) -> int | None:
    value = parameters.get("channel_count", parameters.get("channels"))
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if type(value) is bool or count not in allowed:
        return None
    return count


def _uln_ports(channels: int, relay: bool) -> set[str]:
    ports = {"vmotor", "gnd"}
    for channel in range(1, channels + 1):
        ports.add(f"control_{channel}")
        if relay:
            ports |= {f"relay_{channel}_common", f"relay_{channel}_normally_open", f"relay_{channel}_normally_closed"}
        else:
            ports |= {f"load_{channel}_positive", f"load_{channel}_negative"}
    return ports


def _uln2003(requirement: CircuitRequirement, *, relay: bool) -> LoweringArtifact | None:
    channels = _channels(requirement.parameters, {2, 3, 4} if relay else {2, 4, 8})
    if channels is None:
        return None
    if requirement.exact_part and str(requirement.exact_part).casefold() not in {"uln2003adr", "uln2003a"}:
        return None
    ports = _required_ports(requirement, _uln_ports(channels, relay))
    if ports is None or not _different(ports["vmotor"], ports["gnd"]):
        return None
    if relay:
        # On-board coils are this circuit's internal load, not an external
        # connector contract the architecture must invent and wire.
        negatives = [
            f"__KICRAFT_RELAY__{requirement.id}__COIL_{i}"
            for i in range(1, channels + 1)
        ]
        if set(negatives) & set(ports.values()):
            return None
    else:
        if any(ports[f"load_{i}_positive"] != ports["vmotor"] for i in range(1, channels + 1)):
            return None
        negatives = [ports[f"load_{i}_negative"] for i in range(1, channels + 1)]
    controls = [ports[f"control_{i}"] for i in range(1, channels + 1)]
    voltage = requirement.parameters.get("motor_voltage_v", 5.0)
    if voltage not in ({5.0, 24.0} if relay else {5.0}) or requirement.parameters.get("load_current_a", 0.10) != 0.10:
        return None
    relay_24v = relay and voltage == 24.0
    if len(set(negatives)) != channels or len(set(controls)) != channels or set(negatives) & {ports["vmotor"], ports["gnd"]}:
        return None
    devices = math.ceil(channels / 7)
    groups = [
        _group("driver", "U", "ULN2003ADR", "uln2003adr:ULN2003ADR", "uln2003adr:SOIC-16_L9.9-W3.9-P1.27-LS6.0-BL", quantity=devices, mpn="ULN2003ADR", datasheet=_TI_ULN),
        *([] if relay else [_terminal("load_terminal", channels)]),
        _resistor("input_pulldown", "100k", channels),
        (_group("motor_tvs", "D", "SMBJ26A", "twgmc-smbj26a:SMBJ26A_C726831", "twgmc-smbj26a:SMB_L4.4-W3.6-LS5.3-RD", mpn="SMBJ26A", datasheet=_SMBJ26, sourcing_note="LCSC C726831") if relay_24v else _tvs()),
    ]
    if relay:
        groups += [
            (_group("relay", "K", "SRD-24VDC-SL-C", "srd-24vdc-sl-c:SRD-24VDC-SL-C", "srd-24vdc-sl-c:RELAY-TH_SRD-XXVDC-XL-C", quantity=channels, mpn="SRD-24VDC-SL-C", datasheet=_SONGLE_SRD) if relay_24v else
             _group("relay", "K", "SRD-05VDC-SL-C", "srd-05vdc-sl-c:SRD-05VDC-SL-C", "srd-05vdc-sl-c:RELAY-TH_SRD-XXVDC-XL-C", quantity=channels, mpn="SRD-05VDC-SL-C")),
            _relay_terminal("relay_contact_terminal", channels),
        ]
    # Both supplier symbols use pin 1 for cathode and pin 2 for anode.
    pins: list[LoweringPin] = [LoweringPin(role="motor_tvs", pin="1", net=ports["vmotor"]), LoweringPin(role="motor_tvs", pin="2", net=ports["gnd"])]
    no_connects: list[LoweringNoConnect] = []
    for device in range(devices):
        pins += [LoweringPin(role="driver", index=device, pin="8", net=ports["gnd"]), LoweringPin(role="driver", index=device, pin="9", net=ports["vmotor"])]
        for local in range(1, 8):
            index = device * 7 + local
            inp, out = str(local), str(17 - local)
            if index <= channels:
                pins += [
                    LoweringPin(role="driver", index=device, pin=inp, net=ports[f"control_{index}"]),
                    LoweringPin(role="driver", index=device, pin=out, net=negatives[index - 1]),
                    LoweringPin(role="input_pulldown", index=index - 1, pin="1", net=ports[f"control_{index}"]),
                    LoweringPin(role="input_pulldown", index=index - 1, pin="2", net=ports["gnd"]),
                ]
                if relay:
                    pins += [
                        LoweringPin(role="relay", index=index - 1, pin="1", net=ports["vmotor"]),
                        LoweringPin(role="relay", index=index - 1, pin="4", net=negatives[index - 1]),
                        LoweringPin(role="relay_contact_terminal", index=index - 1, pin="1", net=ports[f"relay_{index}_common"]),
                        LoweringPin(role="relay_contact_terminal", index=index - 1, pin="2", net=ports[f"relay_{index}_normally_open"]),
                        LoweringPin(role="relay_contact_terminal", index=index - 1, pin="3", net=ports[f"relay_{index}_normally_closed"]),
                        LoweringPin(role="relay", index=index - 1, pin="5", net=ports[f"relay_{index}_common"]),
                        LoweringPin(role="relay", index=index - 1, pin="2", net=ports[f"relay_{index}_normally_open"]),
                        LoweringPin(role="relay", index=index - 1, pin="3", net=ports[f"relay_{index}_normally_closed"]),
                    ]
                else:
                    pins += [
                        LoweringPin(role="load_terminal", index=index - 1, pin="1", net=ports["vmotor"]),
                        LoweringPin(role="load_terminal", index=index - 1, pin="2", net=negatives[index - 1]),
                    ]
            else:
                pins.append(LoweringPin(role="driver", index=device, pin=inp, net=ports["gnd"]))
                no_connects.append(LoweringNoConnect(role="driver", index=device, pin=out, reason="unused ULN2003 output is deliberately open"))
    return _artifact(
        "relay-output@2" if relay else "uln2003-actuator@1", requirement, tuple(groups), tuple(pins), no_connects=tuple(no_connects),
        assumptions=((
            "Songle 24 V L coil: 1600 ohm +/-10%, 15 mA nominal, initial operate voltage <=18 V at 20 C. At 20 V rail and 2 V driver drop the initial operate-voltage bound is met; require a 20-26 V rail at the circuit, with temperature, supply tolerance and startup reviewed for the actual installation.",
            "TI specifies ULN2003A VI(on) <=2.4 V at 200 mA and VCE=2 V at 25 C; 2.64 V GPIO high exceeds that test voltage. The 15 mA coil operates below the specified test current, not at the test point: actual GPIO loading, temperature and pull-in require product verification.",
            "COM is VMOTOR; SMBJ26A cathode is VMOTOR and anode GND. Its 26 V stand-off is not a source surge rating: 42.1 V max at 14.3 A 10/1000 us plus the ULN clamp diode's 2 V max at 350 mA is below the 50 V collector rating only at those specified test conditions. Source transient waveform, energy, wiring overshoot and repeated-pulse behavior require review.",
            "Contact load, isolation/clearance, copper, source current budget and assembled thermal behavior are not qualified by this lowering.",
        ) if relay_24v else (
            "ULN2003ADR default: VMOTOR is a distinct 5 V load rail and every output is limited to 0.10 A including inrush/stall; source, copper and die thermal margins require concrete-load review.",
            "COM is physically VMOTOR and the SMBJ5.0A cathode is VMOTOR/anode GND; repeated transient energy, load performance and assembled thermal behavior remain unverified.",
        )),
        calculations=((LoweringCalculation(output="nominal_relay_coil_current_a", equation="24 V / 1600 ohm", inputs={"coil_voltage_v": 24.0, "coil_resistance_ohm": 1600.0, "channels": float(channels)}, chosen_value=f"{channels * .015:g}A nominal aggregate before driver losses", tolerance="nominal at 20 C; not an energized or thermal guarantee"),) if relay_24v else (LoweringCalculation(output="default_load_current_a", equation="0.10 A per independently switched low-side channel", inputs={"channels": float(channels), "channel_current_a": 0.10}, chosen_value=f"{channels * .10:g}A aggregate before logic/clamp losses", tolerance="default circuit declaration; not a silicon or thermal rating"),)),
        named_part_identities=(("ULN2003ADR", "SMBJ26A", "SRD-24VDC-SL-C") if relay_24v else ("ULN2003ADR", "SMBJ5.0A-13-F", *( ("SRD-05VDC-SL-C",) if relay else () ))),
    )


def _tb_ports(channels: int) -> set[str]:
    ports = {"vcc", "vmotor", "gnd", "stby"}
    for channel in range(1, channels + 1):
        ports |= {f"motor_{channel}_in1", f"motor_{channel}_in2", f"motor_{channel}_pwm", f"motor_{channel}_positive", f"motor_{channel}_negative"}
    return ports


def _tb6612(requirement: CircuitRequirement) -> LoweringArtifact | None:
    channels = _channels(requirement.parameters, {2, 4, 8})
    if channels is None or channels % 2:
        return None
    if requirement.exact_part and str(requirement.exact_part).casefold() != "tb6612fng":
        return None
    ports = _required_ports(requirement, _tb_ports(channels))
    if ports is None or not _different(ports["vcc"], ports["vmotor"], ports["gnd"]):
        return None
    controls = [ports[f"motor_{i}_{signal}"] for i in range(1, channels + 1) for signal in ("in1", "in2", "pwm")]
    outputs = [ports[f"motor_{i}_{polarity}"] for i in range(1, channels + 1) for polarity in ("positive", "negative")]
    if len(set(controls)) != len(controls) or len(set(outputs)) != len(outputs) or set(outputs) & {ports["vcc"], ports["vmotor"], ports["gnd"]}:
        return None
    if requirement.parameters.get("motor_voltage_v", 5.0) != 5.0 or requirement.parameters.get("motor_current_a", 0.20) != 0.20:
        return None
    devices = channels // 2
    groups = [
        _group("driver", "U", "TB6612FNG", "tb6612fng:TB6612FNG", "tb6612fng:SSOP24-P-300-0.65A", quantity=devices, mpn="TB6612FNG", datasheet=_TOSHIBA_TB),
        _terminal("motor_terminal", channels),
        _resistor("control_pulldown", "100k", channels * 3),
        _resistor("stby_pulldown", "100k", 1),
        _capacitor("vcc_bulk", "10uF 10V", devices), _capacitor("vcc_decoupling", "100nF 10V", devices),
        _capacitor("vm_bulk", "10uF 16V", devices), _capacitor("vm_decoupling", "100nF 16V", devices), _tvs(),
    ]
    pins: list[LoweringPin] = [
        LoweringPin(role="motor_tvs", pin="1", net=ports["vmotor"]), LoweringPin(role="motor_tvs", pin="2", net=ports["gnd"]),
        LoweringPin(role="stby_pulldown", pin="1", net=ports["stby"]), LoweringPin(role="stby_pulldown", pin="2", net=ports["gnd"]),
    ]
    for device in range(devices):
        a, b = device * 2 + 1, device * 2 + 2
        pins += [
            *[LoweringPin(role="driver", index=device, pin=pin, net=ports[f"motor_{a}_positive"]) for pin in ("1", "2")],
            *[LoweringPin(role="driver", index=device, pin=pin, net=ports["gnd"]) for pin in ("3", "4")],
            *[LoweringPin(role="driver", index=device, pin=pin, net=ports[f"motor_{a}_negative"]) for pin in ("5", "6")],
            *[LoweringPin(role="driver", index=device, pin=pin, net=ports[f"motor_{b}_negative"]) for pin in ("7", "8")],
            *[LoweringPin(role="driver", index=device, pin=pin, net=ports["gnd"]) for pin in ("9", "10")],
            *[LoweringPin(role="driver", index=device, pin=pin, net=ports[f"motor_{b}_positive"]) for pin in ("11", "12")],
            *[LoweringPin(role="driver", index=device, pin=pin, net=ports["vmotor"]) for pin in ("13", "14", "24")],
            LoweringPin(role="driver", index=device, pin="15", net=ports[f"motor_{b}_pwm"]),
            LoweringPin(role="driver", index=device, pin="16", net=ports[f"motor_{b}_in2"]), LoweringPin(role="driver", index=device, pin="17", net=ports[f"motor_{b}_in1"]),
            LoweringPin(role="driver", index=device, pin="18", net=ports["gnd"]), LoweringPin(role="driver", index=device, pin="19", net=ports["stby"]), LoweringPin(role="driver", index=device, pin="20", net=ports["vcc"]),
            LoweringPin(role="driver", index=device, pin="21", net=ports[f"motor_{a}_in1"]), LoweringPin(role="driver", index=device, pin="22", net=ports[f"motor_{a}_in2"]), LoweringPin(role="driver", index=device, pin="23", net=ports[f"motor_{a}_pwm"]),
            LoweringPin(role="motor_terminal", index=a - 1, pin="1", net=ports[f"motor_{a}_positive"]), LoweringPin(role="motor_terminal", index=a - 1, pin="2", net=ports[f"motor_{a}_negative"]),
            LoweringPin(role="motor_terminal", index=b - 1, pin="1", net=ports[f"motor_{b}_positive"]), LoweringPin(role="motor_terminal", index=b - 1, pin="2", net=ports[f"motor_{b}_negative"]),
            *[LoweringPin(role="vcc_bulk", index=device, pin=pin, net=ports["vcc"] if pin == "1" else ports["gnd"]) for pin in ("1", "2")],
            *[LoweringPin(role="vcc_decoupling", index=device, pin=pin, net=ports["vcc"] if pin == "1" else ports["gnd"]) for pin in ("1", "2")],
            *[LoweringPin(role="vm_bulk", index=device, pin=pin, net=ports["vmotor"] if pin == "1" else ports["gnd"]) for pin in ("1", "2")],
            *[LoweringPin(role="vm_decoupling", index=device, pin=pin, net=ports["vmotor"] if pin == "1" else ports["gnd"]) for pin in ("1", "2")],
        ]
        for channel in (a, b):
            for signal in ("in1", "in2", "pwm"):
                pulldown = (channel - 1) * 3 + ("in1", "in2", "pwm").index(signal)
                pins += [LoweringPin(role="control_pulldown", index=pulldown, pin="1", net=ports[f"motor_{channel}_{signal}"]), LoweringPin(role="control_pulldown", index=pulldown, pin="2", net=ports["gnd"])]
    return _artifact(
        "tb6612fng-dual-h-bridge@1", requirement, tuple(groups), tuple(pins),
        assumptions=(
            "TB6612FNG default: VCC is 3.3 V logic, VMOTOR is a distinct protected 5 V load rail, and every motor is limited to 0.20 A continuous and stall current.",
            "STBY has a physical 100 kOhm pulldown, so reset is standby/high impedance. Firmware must assert STBY deliberately; braking, thermal, regenerative energy and assembled performance remain unverified.",
        ),
        calculations=(LoweringCalculation(output="default_motor_rail_current_a", equation="0.20 A per independently bridge-driven motor", inputs={"channels": float(channels), "motor_current_a": .20}, chosen_value=f"{channels * .20:g}A before clamp/converter losses", tolerance="default circuit declaration; not a TB6612 thermal rating"),),
        named_part_identities=("TB6612FNG", "SMBJ5.0A-13-F"),
    )


def _socket_ports() -> set[str]:
    return {"vmotor", "vdd", "gnd", "enable_n", "ms1", "ms2", "ms3", "reset_n", "sleep_n", "step", "dir", "coil_1a", "coil_1b", "coil_2a", "coil_2b"}


def _a4988_socket(requirement: CircuitRequirement) -> LoweringArtifact | None:
    if requirement.exact_part and str(requirement.exact_part).casefold() not in {"pololu 1182", "a4988 driver socket", "a4988 carrier socket", "a4988-carrier-socket"}:
        return None
    ports = _required_ports(requirement, _socket_ports())
    if ports is None or not _different(ports["vmotor"], ports["vdd"], ports["gnd"]):
        return None
    coils = {ports[key] for key in ("coil_1a", "coil_1b", "coil_2a", "coil_2b")}
    if len(coils) != 4 or coils & {ports["vmotor"], ports["vdd"], ports["gnd"]}:
        return None
    groups = [
        _group("carrier_socket", "J", "A4988 carrier socket", "a4988-carrier-socket:A4988_CARRIER_SOCKET", "a4988-carrier-socket:A4988_CARRIER_SOCKET_POLolu_1182", datasheet=_POLOLU_A4988),
        _group("motor_terminal", "J", "WJ126V-5.0-4P", "screw-terminal-5mm-4p:WJ126V-5.0-4P", "screw-terminal-5mm-4p:CONN-TH_4P-P5.00_WJ126V-5.0-4P-1", mpn="WJ126V-5.0-04P-14-00A"),
        _capacitor("vm_bulk", "100uF 35V", 1), _capacitor("vm_decoupling", "100nF 50V", 1),
        _resistor("enable_pullup", "100k", 1), _resistor("control_pulldown", "100k", 5),
    ]
    # The carrier is the A4988 circuit.  This combined native land pattern
    # preserves both 1x08 mating rows at their real 15.24 mm separation.
    socket_pins = {
        "1": "enable_n", "2": "ms1", "3": "ms2", "4": "ms3",
        "5": "reset_n", "6": "sleep_n", "7": "step", "8": "dir",
        "9": "gnd", "10": "vdd", "11": "coil_1b", "12": "coil_1a",
        "13": "coil_2a", "14": "coil_2b", "15": "gnd", "16": "vmotor",
    }
    pins = [
        *[LoweringPin(role="carrier_socket", pin=pin, net=ports[name]) for pin, name in socket_pins.items()],
        LoweringPin(role="motor_terminal", pin="1", net=ports["coil_1a"]), LoweringPin(role="motor_terminal", pin="2", net=ports["coil_1b"]),
        LoweringPin(role="motor_terminal", pin="3", net=ports["coil_2a"]), LoweringPin(role="motor_terminal", pin="4", net=ports["coil_2b"]),
        LoweringPin(role="vm_bulk", pin="1", net=ports["vmotor"]), LoweringPin(role="vm_bulk", pin="2", net=ports["gnd"]), LoweringPin(role="vm_decoupling", pin="1", net=ports["vmotor"]), LoweringPin(role="vm_decoupling", pin="2", net=ports["gnd"]),
        LoweringPin(role="enable_pullup", pin="1", net=ports["vdd"]), LoweringPin(role="enable_pullup", pin="2", net=ports["enable_n"]),
    ]
    for index, name in enumerate(("ms1", "ms2", "ms3", "reset_n", "sleep_n")):
        pins += [LoweringPin(role="control_pulldown", index=index, pin="1", net=ports[name]), LoweringPin(role="control_pulldown", index=index, pin="2", net=ports["gnd"])]
    return _artifact(
        "a4988-carrier-socket@1", requirement, tuple(groups), tuple(pins),
        assumptions=(
            "This is the Pololu 1182-compatible 15.2 x 20.3 mm A4988 carrier socket, not a bare A4988 IC footprint. Carrier current setting, carrier thermal behavior and motor voltage/current are external constraints and are not invented by this socket circuit.",
            "ENABLE is pulled high; RESET and SLEEP are pulled low. The controller must deliberately release reset/sleep and assert its desired enable state after reset.",
        ),
        named_part_identities=("A4988 carrier socket",),
    )


ULN2003_ACTUATOR_LOWERER = RegisteredLowerer(
    "uln2003-actuator@1", frozenset({"uln2003-actuator", "uln2003", "darlington-array", "darlington_array"}), lambda requirement: _uln2003(requirement, relay=False),
    parameter_keys=("channel_count", "channels", "motor_voltage_v", "load_current_a"), port_keys=("vmotor, gnd; control_<n>; load_<n>_positive/load_<n>_negative",), required_port_keys=("vmotor", "gnd"),
    parameter_choices=(("channel_count", (2, 4, 8)), ("channels", (2, 4, 8)), ("motor_voltage_v", (5.0,)), ("load_current_a", (0.10,))), port_directions=(("vmotor", "input"), ("gnd", "bidirectional")), port_patterns=((r"control_[1-8]", "input"), (r"load_[1-8]_positive", "output"), (r"load_[1-8]_negative", "output")), reviewed_exact_part="ULN2003ADR",
)
RELAY_OUTPUT_LOWERER = RegisteredLowerer(
    "relay-output@2", frozenset({"relay-output", "relay_output", "relay-board", "relay_board"}), lambda requirement: _uln2003(requirement, relay=True),
    parameter_keys=("channel_count", "channels", "motor_voltage_v", "load_current_a"), port_keys=("vmotor, gnd; control_<n>; relay_<n>_common/normally_open/normally_closed",), required_port_keys=("vmotor", "gnd"),
    parameter_choices=(("channel_count", (2, 3, 4)), ("channels", (2, 3, 4)), ("motor_voltage_v", (5.0, 24.0)), ("load_current_a", (0.10,))), port_directions=(("vmotor", "input"), ("gnd", "bidirectional")), port_patterns=((r"control_[1-4]", "input"), (r"relay_[1-4]_(?:common|normally_open|normally_closed)", "output")), reviewed_exact_part="ULN2003ADR",
    parameter_constraints=("On-board relay coil supply and distinct driver-to-coil nodes are compiler-owned; do not declare load_<n> ports or extra coil connectors. Bind only the explicitly selected 5 V or 24 V supply, return, independent control inputs and isolated C/NO/NC contact interfaces. The 24 V option requires 20-26 V at the circuit and product-specific transient, temperature, load and thermal review.",),
)
TB6612_LOWERER = RegisteredLowerer(
    "tb6612fng-dual-h-bridge@1", frozenset({"tb6612fng", "tb6612", "dual-dc-motor-driver-tb6612"}), _tb6612,
    parameter_keys=("channel_count", "channels", "motor_voltage_v", "motor_current_a"), port_keys=("vcc, vmotor, gnd, stby; motor_<n>_in1/in2/pwm/positive/negative",), required_port_keys=("vcc", "vmotor", "gnd", "stby"),
    parameter_choices=(("channel_count", (2, 4, 8)), ("channels", (2, 4, 8)), ("motor_voltage_v", (5.0,)), ("motor_current_a", (0.20,))), port_directions=(("vcc", "input"), ("vmotor", "input"), ("gnd", "bidirectional"), ("stby", "input")), port_patterns=((r"motor_[1-8]_(?:in1|in2|pwm)", "input"), (r"motor_[1-8]_(?:positive|negative)", "output")), reviewed_exact_part="TB6612FNG",
)
A4988_CARRIER_SOCKET_LOWERER = RegisteredLowerer(
    "a4988-carrier-socket@1", frozenset({"a4988-driver-socket", "a4988-carrier-socket", "stepper-driver-socket"}), _a4988_socket,
    port_keys=("vmotor, vdd, gnd, enable_n, ms1, ms2, ms3, reset_n, sleep_n, step, dir, coil_1a/1b/2a/2b",), required_port_keys=("vmotor", "vdd", "gnd"),
    port_directions=(("vmotor", "input"), ("vdd", "input"), ("gnd", "bidirectional")), port_patterns=((r"(?:enable_n|ms[1-3]|reset_n|sleep_n|step|dir)", "input"), (r"coil_[12][ab]", "output")), reviewed_exact_part="a4988-carrier-socket",
)

for _lowerer in (ULN2003_ACTUATOR_LOWERER, RELAY_OUTPUT_LOWERER, TB6612_LOWERER, A4988_CARRIER_SOCKET_LOWERER):
    register_lowerer(_lowerer)
