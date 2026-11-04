"""Deterministic, source-backed analog sensor conditioner lowering.

The lowerer owns only circuits whose component identities, operating envelopes and
actual pin maps are reviewed in :mod:`kicraft.design.part_identity`.  It deliberately
has no op-amp enable/reset contact: the requested dual op amps do not provide one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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


_ANALOG_PARTS = {
    "mcp6002-i/sn": ("MCP6002-I/SN", "Amplifier_Operational:MCP6002-xSN", 1.8, 6.0),
    "lm358dr": ("LM358DR", "Amplifier_Operational:LM358", 3.0, 30.0),
    "tlv9062idr": ("TLV9062IDR", "Amplifier_Operational:TLV9062", 1.8, 5.5),
}
_SENSOR_DEFAULTS = {
    "temperature": "MCP9700AT-E/TT",
    "humidity": "SHT30-ARP-B2.5KS",
    "ambient-light": "TEMT6000X01",
    "current-sense": "WSL2512R1000FEA",
    "hall-effect": "DRV5055A3QDBZR",
}
_SENSOR_ALIASES = {"ambient_light": "ambient-light", "current_sense": "current-sense", "hall": "hall-effect", "hall_effect": "hall-effect"}


@dataclass(frozen=True)
class _Channel:
    amp: int
    plus: str
    minus: str
    output: str


def _group(role: str, prefix: str, value: str, symbol: str, footprint: str, *, quantity: int = 1, mpn: str | None = None, datasheet: str | None = None) -> LoweringGroup:
    return LoweringGroup(role=role, reference_prefix=prefix, quantity=quantity, value=value, symbol=symbol, footprint=footprint, mpn=mpn, datasheet=datasheet)


def _resistor(role: str, value: str, quantity: int) -> LoweringGroup:
    return _group(role, "R", value, "Device:R", "Resistor_SMD:R_0603_1608Metric", quantity=quantity)


def _capacitor(role: str, value: str, quantity: int) -> LoweringGroup:
    return _group(role, "C", value, "Device:C", "Capacitor_SMD:C_0603_1608Metric", quantity=quantity)


def _sensor_group(sensor: str, channels: int, sensor_identity: str | None = None) -> LoweringGroup:
    records = {
        "temperature": ("MCP9700AT-E/TT", "Sensor_Temperature:MCP9700Ax-ETT", "Package_TO_SOT_SMD:SOT-23"),
        # Honeywell 009020-1-EN Figure 5 gives the HIH-4010-001's actual
        # 2.54 mm SIP body/lead geometry and -Ve/OUT/+Ve lead order.
        "humidity": ("HIH-4010-001", "hih-4010-001:HIH-4010-001", "hih-4010-001:HIH-4010-001_P2.54mm"),
        "ambient-light": ("TEMT6000X01", "Sensor_Optical:SFH309", "LED_SMD:LED_1206_3216Metric"),
        "hall-effect": ("DRV5055A3QDBZR", "Sensor_Magnetic:DRV5055A3xDBZxQ1", "Package_TO_SOT_SMD:SOT-23"),
        "current-sense": ("WSL2512R1000FEA", "Device:R", "Resistor_SMD:R_2512_6332Metric"),
    }
    mpn, symbol, footprint = records[sensor]
    if sensor == "humidity" and sensor_identity == "SHT30-ARP-B2.5KS":
        mpn = sensor_identity
        symbol = "sht30-arp:SHT30-ARP"
        footprint = "sht30-arp:DFN-8_L2.5-W2.5-P0.50-TL-EP"
    return _group("sensor", "S" if sensor != "current-sense" else "R", mpn, symbol, footprint, quantity=channels, mpn=mpn)




def _ports(requirement: CircuitRequirement, channels: int, sensor: str, power_cycle_isolation: bool) -> dict[str, str] | None:
    required = {"vplus", "vminus", "analog_out_return"}
    required |= {f"sensor_{index}_signal" for index in range(1, channels + 1)}
    required |= {f"sensor_{index}_return" for index in range(1, channels + 1)}
    required |= {f"conditioned_{index}" for index in range(1, channels + 1)}
    required |= {f"analog_out_{index}" for index in range(1, channels + 1)}
    if power_cycle_isolation:
        required |= {"isolation_vdd", "isolation_enable"}
    if sensor == "current-sense":
        required |= {
            f"load_return_{side}_{index}"
            for side in ("in", "out")
            for index in range(1, channels + 1)
        }
    if set(requirement.ports) != required or any(not str(value).strip() for value in requirement.ports.values()):
        return None
    ports = {key: str(value) for key, value in requirement.ports.items()}
    if ports["vminus"] != ports["analog_out_return"]:
        return None
    if power_cycle_isolation and (
        ports["isolation_vdd"] in {ports["vplus"], ports["vminus"]}
        or ports["isolation_enable"] in {ports["vplus"], ports["vminus"], ports["isolation_vdd"]}
    ):
        return None
    for index in range(1, channels + 1):
        if not power_cycle_isolation and ports[f"conditioned_{index}"] != ports[f"analog_out_{index}"]:
            return None
        if power_cycle_isolation and ports[f"conditioned_{index}"] == ports[f"analog_out_{index}"]:
            return None
        if sensor != "current-sense" and ports[f"sensor_{index}_return"] != ports["vminus"]:
            return None
    sensor_signals = {ports[f"sensor_{index}_signal"] for index in range(1, channels + 1)}
    output_nets = {ports[f"conditioned_{index}"] for index in range(1, channels + 1)}
    if len(sensor_signals) != channels or len(output_nets) != channels:
        return None
    if sensor == "current-sense":
        current_inputs = []
        for index in range(1, channels + 1):
            in_key, out_key = f"load_return_in_{index}", f"load_return_out_{index}"
            if (
                ports[out_key] != ports["vminus"]
                or ports[f"sensor_{index}_signal"] != ports[in_key]
                or ports[f"sensor_{index}_return"] != ports[out_key]
            ):
                return None
            current_inputs.append(ports[in_key])
        if len(set(current_inputs)) != channels:
            return None
    return ports


def _channel_pins(index: int) -> _Channel:
    package = index // 2
    return _Channel(package, "3" if index % 2 == 0 else "5", "2" if index % 2 == 0 else "6", "1" if index % 2 == 0 else "7")


def _analog_conditioner(requirement: CircuitRequirement) -> LoweringArtifact | None:
    parameters = requirement.parameters
    sensor = _SENSOR_ALIASES.get(str(parameters.get("sensor_type", parameters.get("topology", "temperature"))).casefold(), str(parameters.get("sensor_type", parameters.get("topology", "temperature"))).casefold())
    try:
        channels = int(parameters.get("channel_count", 2))
        rail = float(parameters.get("analog_rail_v", 5.0))
        load_min = float(parameters.get("output_load_min_ohm", 10_000.0))
        load_cap = float(parameters.get("output_cap_max_pf", 100.0))
        consumer_limit = float(parameters["consumer_input_max_v"]) if "consumer_input_max_v" in parameters else None
    except (TypeError, ValueError):
        return None
    if type(parameters.get("channel_count", 2)) is bool or channels not in {2, 4, 8} or sensor not in _SENSOR_DEFAULTS or rail != 5.0 or load_min < 10_000 or load_cap > 100 or consumer_limit is not None and consumer_limit <= 0:
        return None
    power_cycle_isolation = parameters.get("power_cycle_isolation", False)
    if type(power_cycle_isolation) is not bool:
        return None
    exact_part = str(requirement.exact_part or parameters.get("amplifier_exact_part", "MCP6002-I/SN"))
    exact = exact_part.casefold()
    amplifier = _ANALOG_PARTS.get(exact)
    if amplifier is None or not amplifier[2] <= rail <= amplifier[3]:
        return None
    requested_amplifier = parameters.get("amplifier_exact_part")
    if requirement.exact_part and requested_amplifier and str(requirement.exact_part).casefold() != str(requested_amplifier).casefold():
        return None
    if any(key in parameters for key in ("sensor_transfer", "sensor_range", "output_range_v", "bandwidth_hz")):
        # These defaults have finite documented envelopes, but no alternate
        # transfer/range/bandwidth network has been reviewed. Refuse instead of
        # silently discarding an explicit customer constraint.
        return None
    if "analog_output_connector" in parameters:
        # The conditioner exposes signal ports only. A requested physical output
        # connector is a separate architecture requirement so its type, contact
        # count, and obligation ownership remain visible to the compiler.
        return None
    if "reference_v" in parameters and parameters["reference_v"] not in (0, 0.0, "0", "AGND"):
        return None
    expected_gain = 1.5 if sensor == "temperature" else 24.706 if sensor == "current-sense" and (exact != "tlv9062idr" or consumer_limit is not None and consumer_limit <= 3.3) else 49.5 if sensor == "current-sense" else 1.0
    if "gain" in parameters:
        try:
            if float(parameters["gain"]) != expected_gain:
                return None
        except (TypeError, ValueError):
            return None
    if sensor == "current-sense":
        if str(parameters.get("shunt_identity", "WSL2512R1000FEA")).casefold() != "wsl2512r1000fea":
            return None
        try:
            if float(parameters.get("shunt_resistance_ohm", 0.1)) != 0.1 or float(parameters.get("shunt_power_w", 1.0)) != 1.0:
                return None
        except (TypeError, ValueError):
            return None
    elif any(key in parameters for key in ("shunt_identity", "shunt_resistance_ohm", "shunt_power_w")):
        return None
    reduced_current = sensor == "current-sense" and exact == "tlv9062idr" and consumer_limit is not None and consumer_limit <= 3.3
    high_gain_tlv = sensor == "current-sense" and exact == "tlv9062idr" and not reduced_current
    stated_sensor = str(parameters.get("sensor_identity", _SENSOR_DEFAULTS[sensor]))
    supported_sensors = (
        {"SHT30-ARP-B2.5KS", "HIH-4010-001"} if sensor == "humidity"
        else {_SENSOR_DEFAULTS[sensor]}
    )
    sensor_identity = next(
        (identity for identity in supported_sensors if identity.casefold() == stated_sensor.casefold()),
        None,
    )
    if sensor_identity is None:
        return None
    sht30 = sensor_identity == "SHT30-ARP-B2.5KS"
    reset_button = parameters.get("reset_button", False)
    if type(reset_button) is not bool or reset_button and not sht30:
        return None
    ports = _ports(requirement, channels, sensor, power_cycle_isolation)
    if ports is None:
        return None
    trim = parameters.get("trim", False)
    if type(trim) is not bool:
        return None
    maximum_endpoint = {
        # The -E MCP9700A covers -40..125 C. Its +4 C maximum
        # accuracy error gives 1.79 V at 125 C using the specified transfer.
        # Resistor tolerances and amplifier offset/output swing are not fixed by
        # these generic BOM groups, so only the 5 V rail is a defensible bound.
        "temperature": rail,
        "humidity": (2.62 if trim else 2.5) if sht30 else (2.129 if trim else 2.033),
        "ambient-light": 2.2,
        "hall-effect": 2.743 if trim else 1.92,
        "current-sense": 3.3 if trim and high_gain_tlv else 2.719 if trim else 4.95 if high_gain_tlv else 2.471,
    }[sensor]
    if consumer_limit is not None and maximum_endpoint > consumer_limit:
        return None
    amp_count = channels if trim and high_gain_tlv else math.ceil(channels / 2)
    if trim and high_gain_tlv:
        # A current-sense channel uses A for gain and B to buffer its adjustable
        # output divider, so it deliberately consumes its own dual package.
        amp_count = channels
    switched_capacitance = amp_count * 100e-9
    if sensor == "temperature" or sht30:
        switched_capacitance += channels * 100e-9
    if sht30:
        switched_capacitance += channels * 2e-9
    if sensor == "hall-effect":
        switched_capacitance += channels * 10e-9
    if power_cycle_isolation and switched_capacitance > 1e-6:
        # Include even sensor output capacitors conservatively in the reset
        # domain's total charge; isolator bypasses are on the unswitched rail.
        return None
    part, symbol, _, _ = amplifier
    groups: list[LoweringGroup] = [
        _group("amplifier", "U", exact_part, symbol, "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm", quantity=amp_count, mpn=exact_part),
        _sensor_group(sensor, channels, sensor_identity),
        _capacitor("amplifier_decoupling", "100nF", amp_count),
    ]
    if sensor == "temperature":
        groups.append(_capacitor("sensor_decoupling", "100nF", channels))
    if sht30:
        groups += [
            _capacitor("sensor_decoupling", "100nF", channels),
            _capacitor("humidity_output_filter", "1nF", channels),
            _capacitor("temperature_output_filter", "1nF", channels),
            _resistor("sensor_reset_pullup", "10k", 1),
        ]
        if reset_button:
            groups.append(_group(
                "sensor_reset_button", "SW", "SENSOR RESET",
                "kmr232ngulclfs:KMR232NGULCLFS",
                "kmr232ngulclfs:KMR232NGULCLFS",
                mpn="KMR232NGULCLFS",
            ))
    if power_cycle_isolation:
        groups += [
            _group("output_isolator", "U", "SN74LVC1G66DBVR", "74xGxx:74LVC1G66", "Package_TO_SOT_SMD:SOT-23-5", quantity=channels, mpn="SN74LVC1G66DBVR", datasheet="https://www.ti.com/lit/ds/symlink/sn74lvc1g66.pdf"),
            _capacitor("isolation_decoupling", "100nF", channels),
        ]
        if sensor == "current-sense":
            groups.append(_group("sense_isolator", "U", "SN74LVC1G66DBVR", "74xGxx:74LVC1G66", "Package_TO_SOT_SMD:SOT-23-5", quantity=channels, mpn="SN74LVC1G66DBVR", datasheet="https://www.ti.com/lit/ds/symlink/sn74lvc1g66.pdf"))
            groups.append(_capacitor("sense_isolation_decoupling", "100nF", channels))
    if sensor == "hall-effect":
        groups.append(_capacitor("sensor_decoupling", "10nF", channels))
    if sensor == "temperature":
        groups += [_resistor("gain_ground", "20k", channels), _resistor("gain_feedback", "10k", channels)]
    elif sensor == "humidity":
        groups += [_resistor("divider_top", "100k", channels), _resistor("divider_bottom", "100k", channels)]
    elif sensor in {"ambient-light", "hall-effect"}:
        groups += [_resistor("divider_top", "15k", channels), _resistor("divider_bottom", "10k", channels)]
        if sensor == "ambient-light":
            groups.append(_resistor("emitter_load", "10k", channels))
    else:
        if high_gain_tlv:
            groups += [_resistor("gain_ground", "3.4k", channels), _resistor("gain_feedback", "165k", channels)]
        else:
            groups += [_resistor("gain_ground", "4.02k", channels), _resistor("gain_feedback", "95.3k", channels)]
    if trim:
        groups.append(_group("trim", "RV", "3296W-1-103LF", "trim-pot-3296w-10k:3296W-1-103LF", "trim-pot-3296w-10k:RES-ADJ-TH_3296W", quantity=channels, mpn="3296W-1-103LF"))
        if trim and high_gain_tlv:
            groups += [_resistor("output_divider_top", "10k", channels), _resistor("output_divider_bottom", "10k", channels)]
    pins: list[LoweringPin] = []
    ncs: list[LoweringNoConnect] = []
    reset_net = f"__KICRAFT_ANALOG__{requirement.id}__SENSOR_RESET"
    if sht30:
        pins += [
            LoweringPin(role="sensor_reset_pullup", pin="1", net=ports["vplus"]),
            LoweringPin(role="sensor_reset_pullup", pin="2", net=reset_net),
        ]
        if reset_button:
            pins += [
                LoweringPin(role="sensor_reset_button", pin="1", net=reset_net),
                LoweringPin(role="sensor_reset_button", pin="2", net=ports["vminus"]),
            ]
    for package in range(amp_count):
        pins += [
            LoweringPin(role="amplifier", index=package, pin="8", net=ports["vplus"]),
            LoweringPin(role="amplifier", index=package, pin="4", net=ports["vminus"]),
            LoweringPin(role="amplifier_decoupling", index=package, pin="1", net=ports["vplus"]),
            LoweringPin(role="amplifier_decoupling", index=package, pin="2", net=ports["vminus"]),
        ]
    for index in range(channels):
        number = index + 1
        channel = _channel_pins(index) if not (trim and high_gain_tlv) else _Channel(index, "3", "2", "1")
        raw_external = ports[f"sensor_{number}_signal"]
        raw = (
            f"__KICRAFT_ANALOG__{requirement.id}__SENSE_{number}"
            if power_cycle_isolation and sensor == "current-sense" else raw_external
        )
        out = ports[f"conditioned_{number}"]
        feedback = f"__KICRAFT_ANALOG__{requirement.id}__FB_{number}"
        stage = out if not (trim and high_gain_tlv) else f"__KICRAFT_ANALOG__{requirement.id}__GAIN_{number}"
        if power_cycle_isolation:
            pins += [
                LoweringPin(role="output_isolator", index=index, pin="1", net=out),
                LoweringPin(role="output_isolator", index=index, pin="2", net=ports[f"analog_out_{number}"]),
                LoweringPin(role="output_isolator", index=index, pin="3", net=ports["vminus"]),
                LoweringPin(role="output_isolator", index=index, pin="4", net=ports["isolation_enable"]),
                LoweringPin(role="output_isolator", index=index, pin="5", net=ports["isolation_vdd"]),
                LoweringPin(role="isolation_decoupling", index=index, pin="1", net=ports["isolation_vdd"]),
                LoweringPin(role="isolation_decoupling", index=index, pin="2", net=ports["vminus"]),
            ]
            if sensor == "current-sense":
                pins += [
                    LoweringPin(role="sense_isolator", index=index, pin="1", net=raw_external),
                    LoweringPin(role="sense_isolator", index=index, pin="2", net=raw),
                    LoweringPin(role="sense_isolator", index=index, pin="3", net=ports["vminus"]),
                    LoweringPin(role="sense_isolator", index=index, pin="4", net=ports["isolation_enable"]),
                    LoweringPin(role="sense_isolator", index=index, pin="5", net=ports["isolation_vdd"]),
                    LoweringPin(role="sense_isolation_decoupling", index=index, pin="1", net=ports["isolation_vdd"]),
                    LoweringPin(role="sense_isolation_decoupling", index=index, pin="2", net=ports["vminus"]),
                ]
        feedback_or_output = feedback if sensor in {"temperature", "current-sense"} else stage
        pins += [
            LoweringPin(role="amplifier", index=channel.amp, pin=channel.plus, net=raw if sensor not in {"humidity", "ambient-light", "hall-effect"} else f"__KICRAFT_ANALOG__{requirement.id}__DIV_{number}"),
            LoweringPin(role="amplifier", index=channel.amp, pin=channel.minus, net=feedback_or_output),
            LoweringPin(role="amplifier", index=channel.amp, pin=channel.output, net=stage),
        ]
        if sensor == "temperature":
            pins += [
                LoweringPin(role="sensor", index=index, pin="1", net=ports["vplus"]),
                LoweringPin(role="sensor", index=index, pin="3", net=ports["vminus"]),
                LoweringPin(role="sensor", index=index, pin="2", net=raw),
                LoweringPin(role="sensor_decoupling", index=index, pin="1", net=ports["vplus"]),
                LoweringPin(role="sensor_decoupling", index=index, pin="2", net=ports["vminus"]),
            ]
        elif sensor == "humidity":
            if sht30:
                temperature = f"__KICRAFT_ANALOG__{requirement.id}__TEMPERATURE_{number}"
                pins += [
                    LoweringPin(role="sensor", index=index, pin="1", net=raw),
                    LoweringPin(role="sensor", index=index, pin="4", net=temperature),
                    LoweringPin(role="sensor", index=index, pin="5", net=ports["vplus"]),
                    LoweringPin(role="sensor", index=index, pin="6", net=reset_net),
                    *(LoweringPin(role="sensor", index=index, pin=pin, net=ports["vminus"])
                      for pin in ("2", "3", "7", "8", "9")),
                    LoweringPin(role="sensor_decoupling", index=index, pin="1", net=ports["vplus"]),
                    LoweringPin(role="sensor_decoupling", index=index, pin="2", net=ports["vminus"]),
                    LoweringPin(role="humidity_output_filter", index=index, pin="1", net=raw),
                    LoweringPin(role="humidity_output_filter", index=index, pin="2", net=ports["vminus"]),
                    LoweringPin(role="temperature_output_filter", index=index, pin="1", net=temperature),
                    LoweringPin(role="temperature_output_filter", index=index, pin="2", net=ports["vminus"]),
                ]
            else:
                pins += [LoweringPin(role="sensor", index=index, pin="1", net=ports["vminus"]), LoweringPin(role="sensor", index=index, pin="2", net=raw), LoweringPin(role="sensor", index=index, pin="3", net=ports["vplus"])]
        elif sensor == "ambient-light":
            emitter = raw
            pins += [LoweringPin(role="sensor", index=index, pin="1", net=ports["vplus"]), LoweringPin(role="sensor", index=index, pin="2", net=emitter), LoweringPin(role="emitter_load", index=index, pin="1", net=emitter)]
            if trim:
                trim_net = f"__KICRAFT_ANALOG__{requirement.id}__EMIT_TRIM_{number}"
                pins += [LoweringPin(role="emitter_load", index=index, pin="2", net=trim_net), LoweringPin(role="trim", index=index, pin="1", net=trim_net), LoweringPin(role="trim", index=index, pin="2", net=ports["vminus"]), LoweringPin(role="trim", index=index, pin="3", net=ports["vminus"])]
            else:
                pins.append(LoweringPin(role="emitter_load", index=index, pin="2", net=ports["vminus"]))
        elif sensor == "hall-effect":
            pins += [LoweringPin(role="sensor", index=index, pin="1", net=ports["vplus"]), LoweringPin(role="sensor", index=index, pin="2", net=raw), LoweringPin(role="sensor", index=index, pin="3", net=ports["vminus"]), LoweringPin(role="sensor_decoupling", index=index, pin="1", net=ports["vplus"]), LoweringPin(role="sensor_decoupling", index=index, pin="2", net=ports["vminus"])]
        else:
            pins += [
                LoweringPin(role="sensor", index=index, pin="1", net=ports[f"load_return_in_{number}"]),
                LoweringPin(role="sensor", index=index, pin="2", net=ports[f"load_return_out_{number}"]),
            ]
        if sensor in {"humidity", "ambient-light", "hall-effect"}:
            div = f"__KICRAFT_ANALOG__{requirement.id}__DIV_{number}"
            pins += [LoweringPin(role="divider_top", index=index, pin="1", net=raw), LoweringPin(role="divider_top", index=index, pin="2", net=div), LoweringPin(role="divider_bottom", index=index, pin="1", net=div)]
            if trim and sensor in {"humidity", "hall-effect"}:
                trim_net = f"__KICRAFT_ANALOG__{requirement.id}__DIV_TRIM_{number}"
                pins += [LoweringPin(role="divider_bottom", index=index, pin="2", net=trim_net), LoweringPin(role="trim", index=index, pin="1", net=trim_net), LoweringPin(role="trim", index=index, pin="2", net=ports["vminus"]), LoweringPin(role="trim", index=index, pin="3", net=ports["vminus"])]
            else:
                pins.append(LoweringPin(role="divider_bottom", index=index, pin="2", net=ports["vminus"]))
        if sensor in {"temperature", "current-sense"}:
            pins += [LoweringPin(role="gain_ground", index=index, pin="1", net=feedback), LoweringPin(role="gain_ground", index=index, pin="2", net=ports["vminus"]), LoweringPin(role="gain_feedback", index=index, pin="1", net=stage)]
            if trim and not high_gain_tlv:
                trim_net = f"__KICRAFT_ANALOG__{requirement.id}__FB_TRIM_{number}"
                pins += [LoweringPin(role="gain_feedback", index=index, pin="2", net=trim_net), LoweringPin(role="trim", index=index, pin="1", net=trim_net), LoweringPin(role="trim", index=index, pin="2", net=feedback), LoweringPin(role="trim", index=index, pin="3", net=feedback)]
            else:
                pins.append(LoweringPin(role="gain_feedback", index=index, pin="2", net=feedback))
        if trim and high_gain_tlv:
            div = f"__KICRAFT_ANALOG__{requirement.id}__OUT_DIV_{number}"
            trim_net = f"__KICRAFT_ANALOG__{requirement.id}__OUT_TRIM_{number}"
            pins += [LoweringPin(role="output_divider_top", index=index, pin="1", net=stage), LoweringPin(role="output_divider_top", index=index, pin="2", net=div), LoweringPin(role="output_divider_bottom", index=index, pin="1", net=div), LoweringPin(role="output_divider_bottom", index=index, pin="2", net=trim_net), LoweringPin(role="trim", index=index, pin="1", net=trim_net), LoweringPin(role="trim", index=index, pin="2", net=ports["vminus"]), LoweringPin(role="trim", index=index, pin="3", net=ports["vminus"]), LoweringPin(role="amplifier", index=channel.amp, pin="5", net=div), LoweringPin(role="amplifier", index=channel.amp, pin="6", net=out), LoweringPin(role="amplifier", index=channel.amp, pin="7", net=out)]
    used = {(pin.index, pin.pin) for pin in pins if pin.role == "amplifier"}
    for package in range(amp_count):
        for plus, minus, output in (("3", "2", "1"), ("5", "6", "7")):
            if (package, output) not in used:
                idle = f"__KICRAFT_ANALOG__{requirement.id}__IDLE_{package}_{output}"
                pins += [LoweringPin(role="amplifier", index=package, pin=plus, net=ports["vminus"]), LoweringPin(role="amplifier", index=package, pin=minus, net=idle), LoweringPin(role="amplifier", index=package, pin=output, net=idle)]
    endpoint = {"temperature": 2.625, "humidity": 2.033, "ambient-light": 0.143, "hall-effect": 1.92, "current-sense": 2.475 if trim and high_gain_tlv else 4.95 if high_gain_tlv else 2.471}[sensor]
    if sht30:
        endpoint = 2.25
    return _artifact(
        "analog-conditioner@1",
        requirement,
        tuple(groups),
        tuple(pins),
        no_connects=tuple(ncs),
        assumptions=(
            f"{sensor_identity} is the documented selected sensor; physical accuracy, calibration, noise and assembled performance remain unverified.",
            "OUT<n> supports a >=10 kOhm load to AGND and <=100 pF direct capacitance; no ADC or cable load is implied.",
            "MCP6002, LM358 and TLV9062 have no reset/enable pin; requested reset belongs to a documented controller or the actual sensor reset contact.",
            "The conditioner owns no board-edge output connector. Each analog_out_<n> port must bind to a declared consumer or an architecture-owned connector contact so the requested interface remains independently verifiable.",
            "With power_cycle_isolation, each external output is gated by an unswitched SN74LVC1G66DBVR; current-sense inputs are gated too. isolation_enable must be the independently powered supervisor RESET release regenerated through an unswitched SN74LVC1G17 Schmitt buffer, never raw RESET or power-switch EN. Every isolator has its own 100 nF unswitched-supply bypass. The external interface is limited to 0..5 V.",
            "Each amplifier and MCP9700 or SHT30 sensor needs its own 100 nF bypass placed close to its supply pins; delivered pad connectivity alone does not establish this board-stage placement requirement.",
            "MCP9700A -E operation is specified from -40 to 125 C (Microchip DS20001942K pp. 3–4); at 125 C the nominal 1.75 V sensor output becomes 2.625 V at gain 1.5, and the +4 C sensor error gives 1.79 V before gain. Trim raises nominal gain to 2; generic resistor tolerances and amplifier errors are not specified, so the conservative consumer bound is the 5 V supply rail.",
            "The switched reset domain includes amplifier and sensor bypasses and sensor output capacitors; its total must not exceed 1 uF. Isolation bypasses are on the unswitched supply.",
            *((
                "SHT30-ARP humidity output is 10–90% of VDD (0.5–4.5 V at 5 V); the 100k/100k divider and unity-gain buffer give nominal 0.25–2.25 V for 0–100% RH. Both sensor outputs have 1 nF loads and every sensor has 100 nF bypass. reset_button=true shorts the real active-low sensor nRESET to return; it does not reset or power-cycle the amplifier. The 10 kOhm external reset pull-up exceeds the manufacturer's 2 kOhm minimum. Manufactured accuracy, pulse timing and settling remain unverified.",
            ) if sht30 else ()),
        ),
        calculations=(
            LoweringCalculation(output="nominal_output_endpoint_v", equation="documented default sensor transfer and selected feedback/divider topology", inputs={"analog_rail_v": rail, "channel_count": float(channels)}, chosen_value=f"{endpoint:g}V", tolerance="nominal circuit transfer only; physical accuracy/calibration unverified"),
            LoweringCalculation(output="maximum_output_v", equation="documented baseline span including trim and saturation bounds", inputs={"analog_rail_v": rail, "consumer_input_max_v": consumer_limit if consumer_limit is not None else "none"}, chosen_value=f"{maximum_endpoint:g}V", tolerance="consumer compatibility bound; physical accuracy/calibration unverified"),
        ),
        named_part_identities=(part, sensor_identity) + (("SN74LVC1G66DBVR",) if power_cycle_isolation else ()),
    )


def _temt6000_readout(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Lower the reviewed one-channel TEMT6000 emitter-follower readout.

    This is intentionally separate from ``analog-conditioner@1``.  A direct
    phototransistor acquisition path is not an op-amp conditioner, and its
    public ports are the circuit's supply, return, and acquired output only.
    """
    if str(requirement.exact_part or "").casefold() != "temt6000x01":
        return None
    parameters = requirement.parameters
    try:
        rail = float(parameters["analog_rail_v"])
        consumer_limit = float(parameters["consumer_input_max_v"])
        filter_pf = float(parameters.get("output_filter_pf", 1_000.0))
    except (KeyError, TypeError, ValueError):
        return None
    if (
        rail not in {3.3, 5.0}
        or not math.isfinite(consumer_limit)
        or consumer_limit < rail
        or filter_pf != 1_000.0
        or set(parameters) - {
            "analog_rail_v", "consumer_input_max_v", "output_filter_pf", "reference_v",
        }
        or parameters.get("reference_v", 0.0) not in {0, 0.0, "0", "GND", "AGND"}
    ):
        return None
    if set(requirement.ports) != {"vplus", "vminus", "analog_out"}:
        return None
    ports = {key: str(value) for key, value in requirement.ports.items()}
    if (
        not all(ports.values())
        or len(set(ports.values())) != 3
    ):
        return None
    return _artifact(
        "temt6000-readout@1",
        requirement,
        (
            _sensor_group("ambient-light", 1),
            _resistor("emitter_load", "10k", 1),
            _capacitor("output_filter", "1nF", 1),
        ),
        (
            # Vishay TEMT6000X01: collector is pin 1 and emitter is pin 2.
            LoweringPin(role="sensor", pin="1", net=ports["vplus"]),
            LoweringPin(role="sensor", pin="2", net=ports["analog_out"]),
            LoweringPin(role="emitter_load", pin="1", net=ports["analog_out"]),
            LoweringPin(role="emitter_load", pin="2", net=ports["vminus"]),
            LoweringPin(role="output_filter", pin="1", net=ports["analog_out"]),
            LoweringPin(role="output_filter", pin="2", net=ports["vminus"]),
        ),
        assumptions=(
            "Vishay TEMT6000X01 collector (pin 1) is supplied at 3.3 V or 5.0 V; emitter (pin 2) is the acquired output through a 10 kOhm return load.",
            "The 1 nF output-to-return capacitor with the 10 kOhm load is a 15.9 kHz nominal low-pass. It is not a calibration, illuminance-range, or noise-performance guarantee.",
            "The acquired output is bounded by the observed supply rail and requires a consumer with an input maximum at least equal to that rail. An external connector or ADC/MCU acquisition belongs to its own architecture requirement.",
        ),
        calculations=(
            LoweringCalculation(
                output="nominal_output_at_100_lx_v",
                equation="VOUT = IPCE × RLOAD",
                inputs={"ipce_typical_a": 50e-6, "emitter_load_ohm": 10_000.0},
                chosen_value="0.5V",
                tolerance="TEMT6000X01 typical 50 uA photo current at 100 lx, VCE=5 V, 25 C; not a calibrated illuminance transfer",
            ),
            LoweringCalculation(
                output="maximum_output_v",
                equation="VOUT ≤ VPLUS",
                inputs={"analog_rail_v": rail, "consumer_input_max_v": consumer_limit},
                chosen_value=f"{rail:g}V",
                tolerance="phototransistor saturation/output bound; no calibrated illuminance guarantee",
            ),
            LoweringCalculation(
                output="output_filter_cutoff_hz",
                equation="1 / (2π × RLOAD × CFILTER)",
                inputs={"emitter_load_ohm": 10_000.0, "output_filter_f": 1e-9},
                chosen_value="15915Hz",
                tolerance="nominal passive-component value only",
            ),
        ),
        named_part_identities=("TEMT6000X01",),
    )


TEMT6000_READOUT_LOWERER = RegisteredLowerer(
    "temt6000-readout@1",
    frozenset({
        "ambient-light-phototransistor",
        "ambient_light_phototransistor",
        "ambient-light-sensor",
        "ambient_light_sensor",
    }),
    _temt6000_readout,
    parameter_keys=(
        "analog_rail_v", "consumer_input_max_v", "output_filter_pf", "reference_v",
    ),
    port_keys=("vplus", "vminus", "analog_out"),
    required_port_keys=("vplus", "vminus", "analog_out"),
    required_parameter_keys=("analog_rail_v", "consumer_input_max_v"),
    parameter_choices=(
        ("analog_rail_v", (3.3, 5.0)),
        ("output_filter_pf", (1_000.0,)),
        ("reference_v", (0, 0.0, "0", "GND", "AGND")),
    ),
    reference_port_keys=("vminus",),
    port_directions=(
        ("vplus", "input"),
        ("vminus", "bidirectional"),
        ("analog_out", "output"),
    ),
    required_exact_part="TEMT6000X01",
    reviewed_exact_part="TEMT6000X01",
    parameter_constraints=(
        "TEMT6000X01 is the only supported identity. Bind only vplus, vminus and analog_out; "
        "the collector/load/filter nodes are physical internal circuit connections, not substitute "
        "ports. analog_rail_v is exactly 3.3 V or 5.0 V, below the 6 V collector-emitter maximum, "
        "and consumer_input_max_v must be at least analog_rail_v. The output reference is vminus.",
    ),
)


ANALOG_CONDITIONER_LOWERER = RegisteredLowerer(
    "analog-conditioner@1",
    frozenset({"analog-conditioner", "analog_conditioner", "sensor-conditioner", "sensor_conditioner"}),
    _analog_conditioner,
    parameter_keys=("channel_count", "topology", "sensor_type", "sensor_identity", "sensor_transfer", "sensor_range", "amplifier_exact_part", "analog_rail_v", "output_range_v", "bandwidth_hz", "gain", "reference_v", "output_load_min_ohm", "output_cap_max_pf", "consumer_input_max_v", "shunt_identity", "shunt_resistance_ohm", "shunt_power_w", "trim", "power_cycle_isolation", "reset_button"),
    port_keys=("vplus, vminus; analog_out_<n>; power-cycle isolation also isolation_vdd/isolation_enable; current sense also load_return_in_<n>/load_return_out_<n>",),
    required_port_keys=("vplus", "vminus"),
    reference_port_keys=("vminus",),
    parameter_choices=(("channel_count", (2, 4, 8)), ("sensor_type", tuple(_SENSOR_DEFAULTS)), ("topology", tuple(_SENSOR_DEFAULTS)), ("amplifier_exact_part", tuple(part[0] for part in _ANALOG_PARTS.values())), ("analog_rail_v", (5.0,)), ("trim", (False, True)), ("power_cycle_isolation", (False, True)), ("reset_button", (False, True))),
    port_directions=(("vplus", "input"), ("vminus", "bidirectional"), ("isolation_vdd", "input"), ("isolation_enable", "input")),
    port_patterns=((r"analog_out_[1-8]", "output"), (r"load_return_(?:in|out)_[1-8]", "bidirectional")),
    derived_port_patterns=((r"sensor_[1-8]_(?:signal|return)", "bidirectional"), (r"conditioned_[1-8]", "output"), (r"analog_out_return", "bidirectional")),
    reviewed_exact_parts=tuple(part[0] for part in _ANALOG_PARTS.values()),
    parameter_constraints=(
        "Bind analog_out_<n> to the requested consumers for exactly channel_count channels. vplus is the amplifier positive SUPPLY (SOIC-8 pin 8); vminus is its supply RETURN (pin 4), not an inverting signal input. The compiler binds the declared supply and published return; never drive vminus with sensor, trim or reset signals. The compiler owns the on-board sensors, amplifier inputs, feedback networks and internal conditioned nodes; do not wire them as separate external signals. Set trim=true for the reviewed on-board trimmer instead of driving a supply pin from an external trimmer. With power_cycle_isolation, isolation_vdd is the unswitched 5 V rail and isolation_enable is the reset-supervisor release that stays low until the switched rail has recovered. Current sensing additionally binds a distinct low-side Kelvin load_return_in_<n> and shared-return load_return_out_<n> for every channel; parallel shunts are rejected. The analog output connector is an architecture-owned connector requirement, never an analog-conditioner parameter.",
        "Only the reviewed MCP6002-I/SN, LM358DR and TLV9062IDR SOIC-8 amplifier identities are supported at the fixed 5 V analog rail. Named sensors, output spans, bandwidths, and transfers outside the documented default circuits are refused rather than guessed.",
        "Generic humidity uses sourceable SHT30-ARP-B2.5KS analog sensors, not I2C SHT30-DIS. An explicit HIH-4010-001 remains exact. For a requested humidity reset button set reset_button=true on this conditioner: the compiler owns one physical button and a 10k pull-up on the sensors' actual active-low nRESET. No separate power-cycle/reset-switch requirement is needed for that sensor reset. This option is refused for sensors without reviewed reset contacts.",
    ),
)


for _lowerer in (ANALOG_CONDITIONER_LOWERER, TEMT6000_READOUT_LOWERER):
    register_lowerer(_lowerer)
