"""Fail-closed board-artifact observations for ``analog-conditioner@1``.

This module never trusts recipe roles or net labels.  It derives its facts only
from delivered footprint pads reconciled to reviewed identity records.
"""
from __future__ import annotations
import math
import re
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any


from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.validation import _capacitance_farads, _resistance_ohms
from kicraft.eval.application_pin_evidence import reviewed_application_pins
from kicraft.eval.data_logging_artifact_evidence import i2c_adc_input_nets


_SENSOR_OUTPUT = {
    "mcp9700at-e/tt": "output",
    "hih-4010-001": "output",
    "sht30-arp-b2.5ks": "output",
    "temt6000x01": "emitter",
    "drv5055a3qdbzr": "output",
}
_ENDPOINT_V = {
    "mcp9700at-e/tt": 2.625,
    "hih-4010-001": 2.033,
    "sht30-arp-b2.5ks": 2.25,
    "temt6000x01": 0.143,
    "drv5055a3qdbzr": 1.92,
}
_AMPLIFIERS = {"mcp6002-i/sn", "lm358dr", "tlv9062idr"}


def _record(part: dict[str, Any]):
    return physical_inventory_record(
        mpn=part.get("mpn"), symbol=part.get("symbol"), footprint=part.get("footprint"),
        datasheet=part.get("datasheet"), sourcing_note=part.get("sourcing_note"),
    )


def _net(ref: str, pin: str, pad_nets: dict[tuple[str, str], str]) -> str | None:
    return pad_nets.get((ref, pin))


def _resistor_graph(parts: list[dict[str, Any]], pad_nets: dict[tuple[str, str], str], records: dict[str, Any]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        ref = str(part.get("ref") or "")
        record = records.get(ref)
        if record is not None and record.identity == "3296w-1-103lf":
            left, right = _net(ref, "1", pad_nets), _net(ref, "2", pad_nets)
            if left and right and left != right and _net(ref, "3", pad_nets) == right:
                graph[left].add(right)
                graph[right].add(left)
            continue
        if str(part.get("symbol")) not in {"Device:R", "Device:R_Small"}:
            continue
        pads = [net for (candidate, _), net in pad_nets.items() if candidate == ref]
        if len(pads) != 2 or pads[0] == pads[1] or _resistance_ohms(str(part.get("value") or "")) is None:
            continue
        graph[pads[0]].add(pads[1]); graph[pads[1]].add(pads[0])
    return graph

def _resistor_between(
    parts: list[dict[str, Any]],
    pad_nets: dict[tuple[str, str], str],
    left: str | None,
    right: str | None,
    value_ohms: float,
) -> bool:
    if not left or not right:
        return False
    for part in parts:
        if str(part.get("symbol")) not in {"Device:R", "Device:R_Small"}:
            continue
        ref = str(part.get("ref") or "")
        nets = {_net(ref, "1", pad_nets), _net(ref, "2", pad_nets)}
        resistance = _resistance_ohms(str(part.get("value") or ""))
        if nets == {left, right} and resistance is not None and abs(resistance - value_ohms) <= value_ohms * 0.001:
            return True
    return False

def _series_trim_resistor_between(
    parts: list[dict[str, Any]],
    records: dict[str, Any],
    pad_nets: dict[tuple[str, str], str],
    left: str | None,
    right: str | None,
    value_ohms: float,
) -> bool:
    """Require the complete resistor/rheostat path, not a dangling resistor."""
    if not left or not right or left == right:
        return False
    for part in parts:
        ref = str(part.get("ref") or "")
        record = records.get(ref)
        if record is None or record.identity != "3296w-1-103lf":
            continue
        junction = _net(ref, "1", pad_nets)
        if (
            junction and junction not in {left, right}
            and _net(ref, "2", pad_nets) == _net(ref, "3", pad_nets) == right
            and _resistor_between(parts, pad_nets, left, junction, value_ohms)
        ):
            return True
    return False


def _value_topology_ok(
    parts: list[dict[str, Any]],
    sensors: list[dict[str, Any]],
    records: dict[str, Any],
    amplifiers: list[dict[str, Any]],
    pad_nets: dict[tuple[str, str], str],
    return_net: str,
    trim_present: bool,
    signal_pairs: set[frozenset[str]],
) -> bool:
    """Check only documented resistor endpoints/values, never recipe labels."""
    channels = [
        (_net(str(amp["ref"]), out, pad_nets), _net(str(amp["ref"]), minus, pad_nets), _net(str(amp["ref"]), plus, pad_nets))
        for amp in amplifiers
        for out, minus, plus in (("1", "2", "3"), ("7", "6", "5"))
    ]
    for sensor in sensors:
        record = records[str(sensor["ref"])]
        if record.identity == "wsl2512r1000fea":
            high = _net(str(sensor["ref"]), "1", pad_nets)
            candidates = [
                (out, minus) for out, minus, plus in channels
                if high and plus and (plus == high or frozenset((high, plus)) in signal_pairs)
            ]
            if not candidates:
                return False
            high_gain = any(_resistor_between(parts, pad_nets, out, minus, 165_000.0) and _resistor_between(parts, pad_nets, minus, return_net, 3_400.0) for out, minus in candidates)
            reduced_gain = any(
                (
                    _series_trim_resistor_between(parts, records, pad_nets, out, minus, 95_300.0)
                    if trim_present else _resistor_between(parts, pad_nets, out, minus, 95_300.0)
                )
                and _resistor_between(parts, pad_nets, minus, return_net, 4_020.0)
                for out, minus in candidates
            )
            if not (high_gain or reduced_gain):
                return False
            continue
        source_pin = record.port_pins.get(_SENSOR_OUTPUT[record.identity])
        source = _net(str(sensor["ref"]), str(source_pin), pad_nets) if source_pin else None
        candidates = [(out, minus, plus) for out, minus, plus in channels if plus == source]
        if record.identity == "mcp9700at-e/tt":
            if not any((_series_trim_resistor_between(parts, records, pad_nets, out, minus, 10_000.0) if trim_present else _resistor_between(parts, pad_nets, out, minus, 10_000.0)) and _resistor_between(parts, pad_nets, minus, return_net, 20_000.0) for out, minus, _ in candidates):
                return False
        elif record.identity == "temt6000x01":
            # The emitter load is distinct from the 15k/10k attenuator.
            if not (_series_trim_resistor_between(parts, records, pad_nets, source, return_net, 10_000.0) if trim_present else _resistor_between(parts, pad_nets, source, return_net, 10_000.0)):
                return False
            if not any(_resistor_between(parts, pad_nets, source, plus, 15_000.0) and _resistor_between(parts, pad_nets, plus, return_net, 10_000.0) for _, _, plus in channels):
                return False
        else:
            humidity = record.identity in {"hih-4010-001", "sht30-arp-b2.5ks"}
            expected = 100_000.0 if humidity else 15_000.0
            lower = 100_000.0 if humidity else 10_000.0
            if not any(_resistor_between(parts, pad_nets, source, plus, expected) and (_series_trim_resistor_between(parts, records, pad_nets, plus, return_net, lower) if trim_present else _resistor_between(parts, pad_nets, plus, return_net, lower)) for _, _, plus in channels):
                return False
    return True


def _reaches(graph: dict[str, set[str]], start: str | None, end: str | None, maximum: int = 3) -> bool:
    if not start or not end:
        return False
    todo = deque([(start, 0)]); seen = {start}
    while todo:
        node, depth = todo.popleft()
        if node == end:
            return True
        if depth == maximum:
            continue
        for next_node in graph.get(node, ()):
            if next_node not in seen:
                seen.add(next_node); todo.append((next_node, depth + 1))
    return False

def _capacitors_between(
    parts: list[dict[str, Any]],
    pad_nets: dict[tuple[str, str], str],
    left: str,
    right: str,
    minimum: float,
) -> int:
    return sum(
        1
        for part in parts
        if str(part.get("symbol")) in {"Device:C", "Device:C_Small", "Device:C_Polarized"}
        and {
            _net(str(part.get("ref") or ""), "1", pad_nets),
            _net(str(part.get("ref") or ""), "2", pad_nets),
        } == {left, right}
        and (value := _capacitance_farads(str(part.get("value") or ""))) is not None
        and value >= minimum
    )



def _capacitor_between_exact(
    parts: list[dict[str, Any]],
    pad_nets: dict[tuple[str, str], str],
    left: str,
    right: str,
    value: float,
) -> bool:
    return any(
        str(part.get("symbol")) in {"Device:C", "Device:C_Small", "Device:C_Polarized"}
        and {
            _net(str(part.get("ref") or ""), "1", pad_nets),
            _net(str(part.get("ref") or ""), "2", pad_nets),
        } == {left, right}
        and (capacitance := _capacitance_farads(str(part.get("value") or ""))) is not None
        and math.isclose(capacitance, value, rel_tol=0.001)
        for part in parts
    )

def _all_pads(ref: str, pad_nets: dict[tuple[str, str], str]) -> list[tuple[str, str]]:
    return sorted(
        (pin, net)
        for (candidate, pin), net in pad_nets.items()
        if candidate == ref
    )


def _external_signal_nets(
    parts: list[dict[str, Any]],
    records: dict[str, Any],
    pad_nets: dict[tuple[str, str], str],
    return_net: str,
) -> set[str]:
    """Return only delivered signal contacts of reviewed connector hardware."""
    nets: set[str] = set()
    for part in parts:
        record = records.get(str(part.get("ref")))
        if record is None:
            continue
        features = record.physical_features
        if not (
            "header" in features
            or "screw-clamp-terminal" in features
            or "screw-terminal" in features
            or "terminal-block" in features
            or any("connector" in feature for feature in features)
            or record.family.endswith("-connector")
        ):
            continue
        nets.update(
            net for _, net in _all_pads(str(part.get("ref") or ""), pad_nets)
            if net != return_net
        )
    return nets


def _adc_consumer_nets(
    state: dict[str, Any] | None,
    parts: list[dict[str, Any]],
    records: dict[str, Any],
    pad_nets: dict[tuple[str, str], str],
) -> set[str]:
    """Bind declared ADC ports to actual reviewed MCU/ADC contacts."""
    architecture = (state or {}).get("architecture")
    if not isinstance(architecture, dict):
        return set()
    requirements = architecture.get("requirements")
    if not isinstance(requirements, list):
        return set()
    conditioner_nets = {
        str(value)
        for requirement in requirements
        if isinstance(requirement, dict)
        and str(requirement.get("family") or "").replace("_", "-") in {
            "analog-conditioner", "sensor-conditioner",
        }
        and isinstance((ports := requirement.get("ports")), dict)
        for key, value in ports.items()
        if re.fullmatch(r"analog_out_\d+", str(key))
        and isinstance(value, str)
    }
    if not conditioner_nets:
        return set()
    # The delivered ADC contact, not the caller's name for its port, proves
    # capability. A generic GPIO alias may legitimately name an ADC-capable pad.
    adc_pins = reviewed_application_pins("adc")
    for record in records.values():
        if record is not None and {"adc", "microcontroller"} & record.physical_features:
            adc_pins.setdefault(record.identity, set()).update(
                pin for port, pin in record.port_pins.items()
                if re.fullmatch(r"(?:ain|adc|analog_in(?:put)?)_?\d+", port)
            )
    delivered_consumers = {
        net
        for part in parts
        if (record := records.get(str(part.get("ref")))) is not None
        and {"microcontroller", "adc"} & record.physical_features
        for pin, net in _all_pads(str(part.get("ref") or ""), pad_nets)
        if pin in adc_pins.get(record.identity, ())
    }
    return conditioner_nets & delivered_consumers


def _reviewed_signal_switches(
    parts: list[dict[str, Any]],
    records: dict[str, Any],
    pad_nets: dict[tuple[str, str], str],
    return_net: str,
    switched_supply_nets: set[str],
    reset_observed: bool,
) -> list[dict[str, str]]:
    """Only switches covered by the independently observed reset boundary."""
    if not reset_observed:
        return []
    switches = []
    for part in parts:
        record = records.get(str(part.get("ref")))
        if record is None or record.identity != "sn74lvc1g66dbvr":
            continue
        ref = str(part.get("ref") or "")
        pins = {
            port: _net(ref, pin, pad_nets)
            for port, pin in record.port_pins.items()
            if port in {"signal_a", "signal_b", "ground", "control", "vdd"}
        }
        if (
            set(pins) != {"signal_a", "signal_b", "ground", "control", "vdd"}
            or not all(pins.values())
            or len(set(pins.values())) != 5
            or pins["ground"] != return_net
            or pins["vdd"] in switched_supply_nets
        ):
            continue
        switches.append(pins)
    return switches


def _isolated_output_nets(
    parts: list[dict[str, Any]],
    pad_nets: dict[tuple[str, str], str],
    *,
    switches: list[dict[str, str]],
    stage_nets: set[str],
    endpoints: set[str],
    return_net: str,
    current_inputs: set[str],
    current_input_terminals: set[str],
) -> tuple[set[str], bool]:
    """Prove one enabled switch per output and per external sense input."""
    output_switches = [
        pins for pins in switches
        if len({pins["signal_a"], pins["signal_b"]} & stage_nets) == 1
        and len({pins["signal_a"], pins["signal_b"]} & endpoints) == 1
    ]
    output_nets = {
        next(iter({pins["signal_a"], pins["signal_b"]} & endpoints))
        for pins in output_switches
    }
    staged = {
        next(iter({pins["signal_a"], pins["signal_b"]} & stage_nets))
        for pins in output_switches
    }
    required_switches = list(output_switches)
    if current_inputs:
        sense_switches = [
            pins for pins in switches
            if len({pins["signal_a"], pins["signal_b"]} & current_inputs) == 1
            and len({pins["signal_a"], pins["signal_b"]} & current_input_terminals) == 1
        ]
        sensed = {
            next(iter({pins["signal_a"], pins["signal_b"]} & current_inputs))
            for pins in sense_switches
        }
        if len(sense_switches) != len(current_inputs) or len(sensed) != len(current_inputs):
            return set(), False
        required_switches += sense_switches
    if (
        len(output_switches) != len(stage_nets)
        or staged != stage_nets
        or len(output_nets) != len(stage_nets)
        or not required_switches
    ):
        return set(), False
    controls = {pins["control"] for pins in required_switches}
    vdds = {pins["vdd"] for pins in required_switches}
    if len(controls) != 1 or len(vdds) != 1:
        return set(), False
    vdd = next(iter(vdds))
    if _capacitors_between(parts, pad_nets, vdd, return_net, 100e-9) < len(required_switches):
        return set(), False
    return output_nets, True


def _conditioner_gates(
    sensors: list[dict[str, Any]],
    records: dict[str, Any],
    *,
    powered: bool,
    output_ok: bool,
    feedback_ok: bool,
    sensor_path_ok: bool,
    trim_parts: list[dict[str, Any]],
    trim_ok: bool,
    current_input_ok: bool,
) -> tuple[dict[str, str], dict[str, int]]:
    """Publish only requested-function evidence backed by complete channels.

    The named function is bound to the delivered reviewed sensor identity, not
    to the conditioner recipe, its net names, or the amplifier identity.
    """
    counts: dict[str, int] = defaultdict(int)
    for sensor in sensors:
        record = records.get(str(sensor.get("ref")))
        if record is not None:
            counts[record.identity] += 1
    complete = all((
        powered,
        output_ok,
        feedback_ok,
        sensor_path_ok,
        trim_ok,
    ))
    gates: dict[str, str] = {}

    def publish(identity: str, *names: str) -> None:
        if complete and counts.get(identity, 0):
            gates.update({name: "pass" for name in names})

    publish(
        "mcp9700at-e/tt",
        "external_temperature_acquisition_function",
        "external_temperature_conditioning_function",
        "external_temperature_front_end_function",
        "external_temperature_sensor_function",
    )
    publish(
        "hih-4010-001",
        "external_humidity_conditioning_function",
        "external_humidity_front_end_function",
    )
    publish(
        "sht30-arp-b2.5ks",
        "external_humidity_conditioning_function",
        "external_humidity_front_end_function",
    )
    publish(
        "temt6000x01",
        "external_ambient_light_front_end_function",
        "external_ambient_light_node_function",
    )
    publish(
        "drv5055a3qdbzr",
        "external_hall_effect_conditioning_function",
        "external_hall_effect_sensor_function",
    )

    if complete and counts.get("mcp9700at-e/tt") == 4:
        gates["external_temperature_channel_count_4"] = "pass"
    if complete and (
        counts.get("hih-4010-001") == 8 or counts.get("sht30-arp-b2.5ks") == 8
    ):
        gates["external_humidity_channel_count_8"] = "pass"
    if complete and counts.get("temt6000x01") == 4:
        gates["external_ambient_light_channel_count_4"] = "pass"
    if complete and counts.get("temt6000x01") == 8:
        gates["external_ambient_light_channel_count_8"] = "pass"

    current_complete = complete and current_input_ok and counts.get("wsl2512r1000fea", 0) > 0
    if current_complete:
        gates.update({
            "external_current_sense_amplifier_function": "pass",
            "external_current_sense_conditioning_function": "pass",
            "external_current_sense_front_end_function": "pass",
            "external_current_sense_sensor_function": "pass",
        })
    if current_complete and counts["wsl2512r1000fea"] == 2:
        gates["external_current_sense_channel_count_2"] = "pass"
    if current_complete and trim_parts:
        gates["external_trim_potentiometer_function"] = "pass"
    return gates, dict(counts)


def _output_loading_ok(
    parts: list[dict[str, Any]], pad_nets: dict[tuple[str, str], str],
    outputs: set[str], return_net: str, supplies: set[str],
) -> bool:
    """Bound delivered passive loads; external cable/consumer loading is assumed.

    Count every capacitor attached to an output conservatively. Resistive paths
    to either supply are counted in parallel, including the feedback path.
    """
    edges: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    caps: dict[str, float] = defaultdict(float)
    for part in parts:
        ref = str(part.get("ref") or "")
        left, right = _net(ref, "1", pad_nets), _net(ref, "2", pad_nets)
        if not left or not right or left == right:
            continue
        if (record := _record(part)) is not None and record.identity == "3296w-1-103lf":
            if _net(ref, "3", pad_nets) == right:
                # A rheostat may be adjusted to zero ohms.
                edges[left].append((right, 0.0, ref))
                edges[right].append((left, 0.0, ref))
            continue
        symbol = str(part.get("symbol") or "")
        if symbol in {"Device:C", "Device:C_Small", "Device:C_Polarized"}:
            value = _capacitance_farads(str(part.get("value") or ""))
            if value is None:
                if left in outputs or right in outputs:
                    return False
            else:
                caps[left] += value
                caps[right] += value
        elif symbol in {"Device:R", "Device:R_Small"}:
            value = _resistance_ohms(str(part.get("value") or ""))
            if value is None or value <= 0:
                if left in outputs or right in outputs:
                    return False
                continue
            edges[left].append((right, value, ref))
            edges[right].append((left, value, ref))
    for output in outputs:
        reachable = {output}
        pending_nodes = [output]
        while pending_nodes:
            node = pending_nodes.pop()
            for other, _, _ in edges.get(node, ()):
                if other not in reachable and other not in supplies | {return_net}:
                    reachable.add(other)
                    pending_nodes.append(other)
        if sum(caps[node] for node in reachable) > 100e-12:
            return False
        conductance = 0.0
        pending = [(output, 0.0, frozenset({output}))]
        while pending:
            node, resistance, seen = pending.pop()
            if node in supplies | {return_net}:
                if resistance == 0:
                    return False
                conductance += 1.0 / resistance
                continue
            for other, value, _ in edges.get(node, ()):
                if other not in seen:
                    pending.append((other, resistance + value, seen | {other}))
        if conductance > 1 / 10_000:
            return False
    return True


def analog_conditioner_facts(
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
    *,
    supply_voltage: Callable[[str, str], float | None],
    state: dict[str, Any] | None = None,
    reset_observed: bool = False,
) -> dict[str, Any]:
    """Return only fully evidenced conditioner facts from actual delivered pads."""
    del names  # Reviewed records, rather than rendered symbol names, own these pins.
    records = {str(part.get("ref")): _record(part) for part in parts}
    amplifiers = [
        part for part in parts
        if (record := records.get(str(part.get("ref"))))
        and record.identity in _AMPLIFIERS
    ]
    sensors = [
        part for part in parts
        if (record := records.get(str(part.get("ref"))))
        and record.identity in {*_SENSOR_OUTPUT, "wsl2512r1000fea"}
    ]
    if not amplifiers or not sensors:
        return {}
    return_nets = {_net(str(part.get("ref")), "4", pad_nets) for part in amplifiers}
    if len(return_nets) != 1 or None in return_nets:
        return {}
    return_net = next(iter(return_nets))
    assert return_net is not None
    graph = _resistor_graph(parts, pad_nets, records)
    # Shared supply/return paths do not carry a sensor's measurement signal.
    for blocked in {return_net, *(_net(str(amp["ref"]), "8", pad_nets) for amp in amplifiers)}:
        graph.pop(blocked, None)
        for neighbors in graph.values():
            neighbors.discard(blocked)
    channels = [
        (_net(str(amp["ref"]), out, pad_nets), _net(str(amp["ref"]), minus, pad_nets), _net(str(amp["ref"]), plus, pad_nets))
        for amp in amplifiers
        for out, minus, plus in (("1", "2", "3"), ("7", "6", "5"))
    ]
    powered = True
    amp_supply_nets: set[str] = set()
    for amplifier in amplifiers:
        ref = str(amplifier["ref"])
        vplus, vminus = _net(ref, "8", pad_nets), _net(ref, "4", pad_nets)
        if (
            not vplus
            or vminus != return_net
            or vplus == vminus
            or supply_voltage(vplus, vminus) != 5.0
        ):
            powered = False
            continue
        amp_supply_nets.add(vplus)
    # Each delivered amplifier and each sensor requiring a bypass consumes a
    # distinct capacitor; pad nets alone do not establish physical proximity.
    for supply in amp_supply_nets:
        required = sum(_net(str(amp["ref"]), "8", pad_nets) == supply for amp in amplifiers)
        required += sum(
            records[str(sensor["ref"])].identity in {"mcp9700at-e/tt", "sht30-arp-b2.5ks"}
            and _net(str(sensor["ref"]), str(records[str(sensor["ref"])].port_pins.get("vdd") or records[str(sensor["ref"])].port_pins.get("vcc")), pad_nets) == supply
            for sensor in sensors
        )
        if _capacitors_between(parts, pad_nets, supply, return_net, 100e-9) < required:
            powered = False
    switches = _reviewed_signal_switches(
        parts, records, pad_nets, return_net, amp_supply_nets, reset_observed
    )
    signal_pairs = {
        frozenset((pins["signal_a"], pins["signal_b"])) for pins in switches
    }
    terminal_nets = _external_signal_nets(parts, records, pad_nets, return_net)
    sensor_path_ok = True
    sensor_endpoints = []
    stage_nets: set[str] = set()
    current_inputs: set[str] = set()
    for sensor in sensors:
        ref, record = str(sensor["ref"]), records[str(sensor["ref"])]
        if record.identity == "wsl2512r1000fea":
            high, low = _net(ref, "1", pad_nets), _net(ref, "2", pad_nets)
            candidates = [
                (out, minus, plus) for out, minus, plus in channels
                if high and plus and (plus == high or frozenset((high, plus)) in signal_pairs)
            ]
            if high not in terminal_nets or low != return_net or len(candidates) != 1 or not candidates[0][0]:
                sensor_path_ok = False
                continue
            stage = candidates[0][0]
            # In the high-gain trimmed TLV circuit the other half of this
            # package buffers a 10k/(10k + rheostat) output divider.
            amp = next((amp for amp in amplifiers if _net(str(amp["ref"]), "1", pad_nets) == stage), None)
            if amp and records[str(amp["ref"])].identity == "tlv9062idr":
                amp_ref = str(amp["ref"])
                buffered = _net(amp_ref, "7", pad_nets)
                divider = _net(amp_ref, "5", pad_nets)
                if (
                    buffered and buffered == _net(amp_ref, "6", pad_nets)
                    and divider and _resistor_between(parts, pad_nets, stage, divider, 10_000.0)
                    and _series_trim_resistor_between(parts, records, pad_nets, divider, return_net, 10_000.0)
                ):
                    stage = buffered
                elif (
                    _resistor_between(parts, pad_nets, stage, candidates[0][1], 165_000.0)
                    and any(r is not None and r.identity == "3296w-1-103lf" for r in records.values())
                ):
                    sensor_path_ok = False
                    continue
            stage_nets.add(stage)
            current_inputs.add(candidates[0][2])
            sensor_endpoints.append(
                4.95 if any(records[str(amp["ref"])].identity == "tlv9062idr" for amp in amplifiers) else 2.471
            )
            continue
        source_pin = record.port_pins.get(_SENSOR_OUTPUT[record.identity])
        source = _net(ref, str(source_pin), pad_nets) if source_pin else None
        candidates = [
            (out, minus, plus)
            for out, minus, plus in channels
            if _reaches(graph, source, plus)
        ]
        vdd_pin = record.port_pins.get("vdd") or record.port_pins.get("vcc")
        ground_pin = record.port_pins.get("ground")
        vdd = _net(ref, str(vdd_pin), pad_nets) if vdd_pin else None
        ground = _net(ref, str(ground_pin), pad_nets) if ground_pin else None
        supply_ok = (
            (vdd_pin is None and ground_pin is None)
            or (vdd in amp_supply_nets and ground == return_net)
        )
        if record.identity == "temt6000x01":
            collector = _net(ref, str(record.port_pins["collector"]), pad_nets)
            supply_ok = collector in amp_supply_nets
        bypass_ok = (
            record.identity not in {"mcp9700at-e/tt", "sht30-arp-b2.5ks"}
            or bool(vdd and ground and _capacitors_between(parts, pad_nets, vdd, ground, 100e-9))
        )
        if record.identity == "sht30-arp-b2.5ks":
            temperature = _net(ref, "4", pad_nets)
            reset = _net(ref, "6", pad_nets)
            bypass_ok = all((
                bypass_ok,
                all(_net(ref, pin, pad_nets) == return_net for pin in ("2", "3", "7", "8", "9")),
                bool(source and _capacitor_between_exact(parts, pad_nets, source, return_net, 1e-9)),
                bool(temperature and temperature not in {source, vdd, return_net}
                     and _capacitor_between_exact(parts, pad_nets, temperature, return_net, 1e-9)),
                bool(reset and reset not in {source, temperature, vdd, return_net}
                     and _resistor_between(parts, pad_nets, reset, vdd, 10_000.0)),
            ))
        if len(candidates) != 1 or not candidates[0][0] or not supply_ok or not bypass_ok:
            sensor_path_ok = False
            continue
        stage_nets.add(candidates[0][0])
        sensor_endpoints.append(_ENDPOINT_V[record.identity])
    if len(stage_nets) != len(sensors):
        sensor_path_ok = False
    feedback_ok = True
    for output, minus, _ in channels:
        if not output or not minus or not (output == minus or _reaches(graph, output, minus)):
            feedback_ok = False
    trim_parts = [
        part for part in parts
        if (record := records.get(str(part.get("ref"))))
        and record.identity == "3296w-1-103lf"
    ]
    trim_ok = not trim_parts or all(
        _net(str(part["ref"]), "2", pad_nets)
        == _net(str(part["ref"]), "3", pad_nets)
        != _net(str(part["ref"]), "1", pad_nets)
        for part in trim_parts
    )
    value_topology_ok = _value_topology_ok(
        parts, sensors, records, amplifiers, pad_nets, return_net, bool(trim_parts), signal_pairs
    )
    feedback_ok = feedback_ok and value_topology_ok
    endpoints = terminal_nets | _adc_consumer_nets(state, parts, records, pad_nets)
    architecture = (state or {}).get("architecture")
    requirements = (
        architecture.get("requirements")
        if isinstance(architecture, dict)
        and isinstance(architecture.get("requirements"), list)
        else []
    )
    isolation_required = any(
        str(requirement.get("family") or "").replace("_", "-")
        in {"analog-conditioner", "sensor-conditioner"}
        and isinstance(requirement.get("parameters"), dict)
        and requirement["parameters"].get("power_cycle_isolation") is True
        for requirement in requirements
        if isinstance(requirement, dict)
    )
    output_nets, isolation_ok = _isolated_output_nets(
        parts, pad_nets,
        switches=switches,
        stage_nets=stage_nets,
        endpoints=endpoints,
        return_net=return_net,
        current_inputs=current_inputs,
        current_input_terminals=terminal_nets,
    )
    direct_output_ok = len(stage_nets) == len(sensors) and stage_nets <= endpoints
    output_ok = (
        isolation_ok and len(output_nets) == len(sensors)
        if isolation_required or not direct_output_ok
        else direct_output_ok
    )
    if output_ok:
        delivered_outputs = output_nets if isolation_required or not direct_output_ok else stage_nets
        output_ok = _output_loading_ok(
            parts, pad_nets, stage_nets | delivered_outputs, return_net, amp_supply_nets
        )
    current_input_ok = not current_inputs or (
        isolation_ok if isolation_required else current_inputs <= terminal_nets
    )
    current_high_gain = any(
        _resistance_ohms(str(part.get("value") or "")) == 165_000.0
        for part in parts
        if str(part.get("symbol")) in {"Device:R", "Device:R_Small"}
    )
    if current_inputs:
        current_nominal = 2.475 if trim_parts and current_high_gain else 4.95 if current_high_gain else 2.471
        sensor_endpoints = [
            current_nominal if endpoint in {2.471, 4.95} else endpoint
            for endpoint in sensor_endpoints
        ]
    maximum_endpoints = [
        (2.62 if trim_parts else 2.5) if record.identity == "sht30-arp-b2.5ks" else 3.3 if record.identity == "wsl2512r1000fea" and trim_parts and current_high_gain else 4.95 if record.identity == "wsl2512r1000fea" and current_high_gain else 2.719 if record.identity == "wsl2512r1000fea" and trim_parts else 2.471 if record.identity == "wsl2512r1000fea" else 5.0 if record.identity == "mcp9700at-e/tt" else 2.129 if record.identity == "hih-4010-001" and trim_parts else 2.2 if record.identity == "temt6000x01" else 2.743 if record.identity == "drv5055a3qdbzr" and trim_parts else _ENDPOINT_V[record.identity]
        for record in (records[str(sensor["ref"])] for sensor in sensors)
    ]
    gates, sensor_channel_counts = _conditioner_gates(
        sensors, records,
        powered=powered,
        output_ok=output_ok,
        feedback_ok=feedback_ok,
        sensor_path_ok=sensor_path_ok,
        trim_parts=trim_parts,
        trim_ok=trim_ok,
        current_input_ok=current_input_ok,
    )
    facts: dict[str, Any] = {
        "channel_counts": {"analog_conditioner": len(sensors)},
        "analog_sensor_channel_counts": sensor_channel_counts,
        "net_paths": {},
        "gates": gates,
    }
    paths = facts["net_paths"]
    if powered:
        paths["analog_conditioner_power"] = True
    if output_ok:
        paths["analog_conditioner_outputs"] = True
    if feedback_ok:
        paths["analog_conditioner_feedback"] = True
    if sensor_path_ok:
        paths["analog_conditioner_sensor_path"] = True
    if trim_parts and trim_ok:
        paths["analog_conditioner_trim"] = True
    if all((powered, output_ok, feedback_ok, sensor_path_ok, trim_ok, current_input_ok)):
        # Publish physical delivered output pads only after the full signal
        # chain is proven. In an isolated design these are downstream of the
        # reviewed switch; otherwise they are the amplifier output pads.
        facts["analog_output_nets"] = sorted(delivered_outputs)
        facts["analog_output_endpoint_v"] = max(sensor_endpoints)
        facts["analog_output_max_v"] = max(maximum_endpoints)
        facts["analog_output_load_min_ohm"] = 10_000.0
        facts["analog_output_cap_max_pf"] = 100.0
    if current_inputs and sensor_path_ok and current_input_ok:
        paths["analog_current_sense_kelvin"] = True
    return facts


def temt6000_readout_facts(
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
    *,
    mcu_facts: dict[str, Any],
    powered_mcu_refs: set[str],
    supply_voltage: Callable[[str, str], float | None],
) -> dict[str, Any]:
    """Observe direct TEMT6000 acquisition from physical pads, not recipe intent.

    The readout is deliberately narrower than ``analog_conditioner_facts``:
    TEMT6000 collector to a 3.3 V/5 V observed rail, emitter output, a 10 kOhm
    return load, and a 1 nF output filter.  The output must terminate at either
    a powered complete MCU ADC contact or an ADS1115 input on that controller's
    readable I2C bus, in the same supply/reference domain.
    """
    del names
    records = {str(part.get("ref")): _record(part) for part in parts}
    output_limits: dict[str, float] = {}
    complete_refs = set(mcu_facts.get("complete_mcu_refs", ())) & powered_mcu_refs
    i2c_sda = reviewed_application_pins("i2c-sda")
    i2c_scl = reviewed_application_pins("i2c-scl")
    for sensor in parts:
        ref = str(sensor.get("ref") or "")
        record = records.get(ref)
        if record is None or record.identity != "temt6000x01":
            continue
        collector = _net(ref, "1", pad_nets)
        output = _net(ref, "2", pad_nets)
        if not collector or not output or collector == output:
            continue
        for return_net in {
            other
            for part in parts
            if str(part.get("symbol")) in {"Device:R", "Device:R_Small"}
            for left, right in [(
                _net(str(part.get("ref") or ""), "1", pad_nets),
                _net(str(part.get("ref") or ""), "2", pad_nets),
            )]
            for other in ((right,) if left == output else (left,) if right == output else ())
        }:
            if (
                not return_net
                or return_net in {collector, output}
                or not _resistor_between(parts, pad_nets, output, return_net, 10_000.0)
                or not _capacitor_between_exact(parts, pad_nets, output, return_net, 1e-9)
            ):
                continue
            voltage = supply_voltage(collector, return_net)
            if voltage not in {3.3, 5.0}:
                continue
            mcu_consumer = False
            for mcu in parts:
                mcu_ref = str(mcu.get("ref") or "")
                mcu_record = records.get(mcu_ref)
                supply = mcu_facts.get("complete_mcu_supplies", {}).get(mcu_ref)
                if (
                    mcu_ref not in complete_refs
                    or mcu_record is None
                    or not isinstance(supply, dict)
                    or supply.get("rail") != collector
                    or supply.get("ground") != return_net
                ):
                    continue
                adc_pins = set(reviewed_application_pins("adc").get(mcu_record.identity, set()))
                adc_pins.update(
                    pin for port, pin in mcu_record.port_pins.items()
                    if re.fullmatch(r"(?:ain|adc|analog_in(?:put)?)_?\d+", port)
                )
                if output in {_net(mcu_ref, str(pin), pad_nets) for pin in adc_pins}:
                    mcu_consumer = True
                    break
                for sda_pin in i2c_sda.get(mcu_record.identity, set()):
                    for scl_pin in i2c_scl.get(mcu_record.identity, set()):
                        if output in i2c_adc_input_nets(
                            parts, pad_nets, vdd=collector, ground=return_net,
                            sda=_net(mcu_ref, str(sda_pin), pad_nets),
                            scl=_net(mcu_ref, str(scl_pin), pad_nets),
                            supply_voltage=supply_voltage,
                        ):
                            mcu_consumer = True
            if mcu_consumer:
                output_limits[output] = voltage
    facts: dict[str, Any] = {
        "channel_counts": {"ambient_light_acquisition": len(output_limits)},
        "net_paths": {},
        "gates": {},
    }
    if not output_limits:
        return facts
    facts.update(
        {
            "ambient_light_acquisition_nets": sorted(output_limits),
            "ambient_light_output_max_v": max(output_limits.values()),
        }
    )
    facts["net_paths"]["temt6000_acquisition"] = True
    facts["gates"].update(
        {
            "external_ambient_light_acquisition_function": "pass",
            "external_ambient_light_sensor_function": "pass",
            "external_ambient_light_front_end_function": "pass",
            "external_ambient_light_node_function": "pass",
            f"external_ambient_light_channel_count_{len(output_limits)}": "pass",
        }
    )
    return facts
