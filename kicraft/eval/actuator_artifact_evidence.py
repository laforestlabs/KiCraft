"""Fail-closed delivered-board observations for actuator capability circuits.

The inputs are already reconciled physical BOM rows and pad nets.  This module
never accepts a net name, component reference, or recipe selection as evidence.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.fab_export import extract_lcsc_pin
from kicraft.eval.application_pin_evidence import reviewed_application_pins


def _record(part: dict[str, Any]):
    return physical_inventory_record(mpn=part.get("mpn"), symbol=part.get("symbol"), footprint=part.get("footprint"))


def _identity(part: dict[str, Any]) -> str:
    record = _record(part)
    return record.identity if record is not None else ""


def _pin(ref: str, names: dict[tuple[str, str], str], pads: dict[tuple[str, str], str], pattern: str):
    found = [(pin, net) for (candidate, pin), name in names.items() if candidate == ref and re.fullmatch(pattern, name, re.I) and (net := pads.get((candidate, pin)))]
    return found


def _ground(nets: set[str], names: dict[tuple[str, str], str], pads: dict[tuple[str, str], str]) -> set[str]:
    return {net for endpoint, name in names.items() if re.fullmatch(r"(?:GND|PGND|AGND|DGND|0V)", name, re.I) and (net := pads.get(endpoint))} | {net for net in nets if re.fullmatch(r"(?:GND|PGND|AGND|DGND|0V)", net, re.I)}


def _resistor_to_ground(parts, pads, ground):
    result = set()
    for part in parts:
        if str(part.get("symbol")) not in {"Device:R", "Device:R_Small"}:
            continue
        pair = [net for (ref, _), net in pads.items() if ref == part.get("ref")]
        if len(pair) == 2 and len(set(pair)) == 2 and set(pair) & ground:
            result.add(next(iter(set(pair) - ground)))
    return result


def _part_nets(part: dict[str, Any], pads: dict[tuple[str, str], str]) -> set[str]:
    return {
        net for (ref, _), net in pads.items()
        if ref == str(part.get("ref")) and net
    }


def _is_resistor(part: dict[str, Any]) -> bool:
    return str(part.get("symbol")) in {"Device:R", "Device:R_Small"}


def _is_capacitor(part: dict[str, Any]) -> bool:
    return str(part.get("symbol")) in {
        "Device:C", "Device:C_Small", "Device:C_Polarized",
    }


def _resistor_between(parts, pads, first: str, second: str) -> bool:
    return any(
        _is_resistor(part) and _part_nets(part, pads) == {first, second}
        for part in parts
    )


def _caps_between(parts, pads, first: str, second: str) -> set[str]:
    return {
        str(part.get("value") or "").casefold()
        for part in parts
        if _is_capacitor(part) and _part_nets(part, pads) == {first, second}
    }



def _caps_to_ground(parts, pads, rail: str, ground: set[str]) -> set[str]:
    return {
        str(part.get("value") or "").casefold()
        for part in parts
        if _is_capacitor(part)
        and any(_part_nets(part, pads) == {rail, gnd} for gnd in ground)
    }
def _tvs_clamps(parts, pads, rail: str, ground: set[str], voltage: float = 5.0) -> bool:
    """Require a rated reviewed TVS, cathode to rail and anode to return."""
    expected = {5.0: "smbj5.0a-13-f", 24.0: "smbj26a"}.get(voltage)
    for part in parts:
        record = _record(part)
        if record is None or record.identity != expected:
            continue
        pinned = extract_lcsc_pin(str(part.get("sourcing_note") or ""))
        if pinned and pinned.upper() != record.lcsc:
            continue
        standoff = record.operating_limits.get("standoff_voltage_v")
        clamp = record.operating_limits.get("clamp_voltage_max_v")
        if not isinstance(standoff, (int, float)) or not voltage <= standoff:
            continue
        if not isinstance(clamp, (int, float)) or clamp >= 50.0:
            continue
        ref = str(part.get("ref"))
        if (
            pads.get((ref, record.port_pins["anode"])) in ground
            and pads.get((ref, record.port_pins["cathode"])) == rail
        ):
            return True
    return False


def _physical_control_endpoint(
    net: str, parts: list[dict[str, Any]], pads: dict[tuple[str, str], str],
    excluded: set[str], powered_mcu_refs: set[str],
) -> bool:
    """A control cannot terminate only at its pull resistor and driver pad."""
    for part in parts:
        ref = str(part.get("ref"))
        if ref in excluded or _is_resistor(part) or _is_capacitor(part):
            continue
        record = _record(part)
        if record is not None and "microcontroller" in record.physical_features:
            if ref not in powered_mcu_refs:
                continue
            pins = reviewed_application_pins("output").get(record.identity, set())
            if any(pads.get((ref, pin)) == net for pin in pins):
                return True
        elif net in _part_nets(part, pads):
            return True
    return False



def _supply_return_endpoint(
    rail: str, ground: set[str], parts: list[dict[str, Any]],
    pads: dict[tuple[str, str], str], excluded: set[str],
) -> bool:
    """Require a non-passive realized supply interface or converter return path."""
    return any(
        str(part.get("ref")) not in excluded
        and not _is_resistor(part)
        and not _is_capacitor(part)
        and rail in _part_nets(part, pads)
        and bool(_part_nets(part, pads) & ground)
        for part in parts
    )

def _rail_voltage(
    observed_outputs: list[dict[str, Any]] | None, rail: str, ground: set[str],
    minimum: float, maximum: float,
    supply_voltage: Callable[[str, str], float | None] | None = None,
) -> bool:
    """Accept an actual output or an authoritative bound input-interface voltage."""
    if supply_voltage is not None:
        for ground_net in ground:
            voltage = supply_voltage(rail, ground_net)
            if (
                isinstance(voltage, (int, float))
                and minimum <= float(voltage) <= maximum
            ):
                return True
    return any(
        isinstance(output, dict)
        and output.get("output_net") == rail
        and output.get("ground_net") in ground
        and isinstance(output.get("voltage"), (int, float))
        and minimum <= float(output["voltage"]) <= maximum
        for output in (observed_outputs or [])
    )


def _terminal_pairs(parts, pads):
    pairs = []
    for part in parts:
        record = _record(part)
        if record is None or "screw-terminal" not in record.physical_features or len(record.contacts) != 2:
            continue
        nets = [pads.get((str(part.get("ref")), pin)) for pin in record.contacts]
        if all(nets) and nets[0] != nets[1]:
            pairs.append((str(part.get("ref")), nets[0], nets[1]))
    return pairs


def _three_terminal_sets(parts, pads) -> list[tuple[str, set[str]]]:
    sets = []
    for part in parts:
        record = _record(part)
        if (
            record is None
            or "screw-terminal" not in record.physical_features
            or len(record.contacts) != 3
        ):
            continue
        nets = {pads.get((str(part.get("ref")), pin)) for pin in record.contacts}
        if len(nets) == 3 and None not in nets:
            sets.append((str(part.get("ref")), nets))
    return sets


def actuator_circuit_facts(
    parts: list[dict[str, Any]], names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str], *,
    observed_outputs: list[dict[str, Any]] | None = None,
    supply_voltage: Callable[[str, str], float | None] | None = None,
    powered_mcu_refs: set[str] | None = None,
) -> dict[str, Any]:
    """Observe complete delivered actuator circuits without trusting net labels."""
    paths: dict[str, bool] = {}
    counts: dict[str, int] = {}
    metadata: dict[str, Any] = {}
    gates: dict[str, str] = {}
    ground = _ground(set(pad_nets.values()), names, pad_nets)
    pulled_down = _resistor_to_ground(parts, pad_nets, ground)
    powered_mcu_refs = set(powered_mcu_refs or ())

    # A ULN output is useful only when it reaches a distinct physical two-wire
    # load boundary, COM supplies that load, E returns to ground, and the
    # reviewed flyback network is physically oriented across that exact rail.
    uln_channels = 0
    uln_safe = True
    uln_clamp = True
    uln_functional = 0
    for driver in [part for part in parts if _identity(part) == "uln2003adr"]:
        ref = str(driver.get("ref"))
        com = _pin(ref, names, pad_nets, r"COM")
        emitter = _pin(ref, names, pad_nets, r"(?:E|GND)")
        if len(com) != 1 or len(emitter) != 1 or emitter[0][1] not in ground:
            uln_clamp = False
            continue
        driver_outs = _pin(ref, names, pad_nets, r"(?:OUT[1-7]|[1-7]C)")
        matched: list[tuple[str, str, str | None]] = []
        for pin, out_net in driver_outs:
            pair = [entry for entry in _terminal_pairs(parts, pad_nets) if out_net in entry[1:]]
            if len(pair) != 1:
                continue
            _, first, second = pair[0]
            rail = second if first == out_net else first
            if rail != com[0][1]:
                uln_clamp = False
                continue
            ordinal = re.search(r"[1-7]", names.get((ref, pin), ""))
            input_pattern = (
                rf"(?:IN{ordinal.group(0)}|{ordinal.group(0)}B)"
                if ordinal else r"$^"
            )
            input_net = next(
                (net for _, net in _pin(ref, names, pad_nets, input_pattern)), None
            )
            if input_net not in pulled_down:
                uln_safe = False
            matched.append((pin, out_net, input_net))
        if len({net for _, net, _ in matched}) != len(matched):
            uln_clamp = False
        uln_channels += len(matched)
        if not _tvs_clamps(parts, pad_nets, com[0][1], ground):
            uln_clamp = False
        if (
            matched
            and all(
                input_net is not None
                and _physical_control_endpoint(
                    input_net, parts, pad_nets, {ref}, powered_mcu_refs,
                )
                for _, _, input_net in matched
            )
            and _rail_voltage(
                observed_outputs, com[0][1], {emitter[0][1]}, 4.75, 5.25, supply_voltage,
            )
            and _tvs_clamps(parts, pad_nets, com[0][1], ground)
        ):
            uln_functional += len(matched)
    if uln_channels:
        counts["uln2003_inductive_channel"] = uln_channels
        paths["uln2003_inductive_load_path"] = True
        if uln_safe:
            paths["uln2003_safe_default"] = True
        if uln_clamp:
            paths["uln2003_inductive_clamp"] = True
        metadata["uln2003_default_load_current_a"] = .10
        metadata["uln2003_repeated_energy"] = "unverified"

    # A relay is not a two-wire inductive load: prove its actual coil circuit
    # separately and prove that the isolated C/NO/NC contacts end at one
    # physical three-way screw terminal.  This prevents an off-board relay
    # connector or an unrelated terminal from standing in for relay outputs.
    relay_complete = 0
    relay_contact_bindings = 0
    relay_terminals = _three_terminal_sets(parts, pad_nets)
    for relay in [part for part in parts if _identity(part) in {"srd-05vdc-sl-c", "srd-24vdc-sl-c"}]:
        relay_ref = str(relay.get("ref"))
        record = _record(relay)
        if record is None:
            continue
        coil_voltage = record.operating_limits.get("coil_voltage_v")
        if not isinstance(coil_voltage, (int, float)):
            continue
        coil_a = pad_nets.get((relay_ref, record.port_pins["coil_a"]))
        coil_b = pad_nets.get((relay_ref, record.port_pins["coil_b"]))
        contacts = {
            pad_nets.get((relay_ref, record.port_pins[name]))
            for name in ("common", "normally_open", "normally_closed")
        }
        if (
            not coil_a or not coil_b or coil_a == coil_b
            or len(contacts) != 3 or None in contacts
            or not any(nets == contacts for _, nets in relay_terminals)
        ):
            continue
        relay_contact_bindings += 1
        for driver in [part for part in parts if _identity(part) == "uln2003adr"]:
            driver_ref = str(driver.get("ref"))
            com = _pin(driver_ref, names, pad_nets, r"COM")
            emitter = _pin(driver_ref, names, pad_nets, r"(?:E|GND)")
            if (
                len(com) != 1 or len(emitter) != 1
                or emitter[0][1] not in ground or com[0][1] != coil_a
                or not _tvs_clamps(parts, pad_nets, coil_a, ground, float(coil_voltage))
                or not _rail_voltage(
                    observed_outputs, coil_a, {emitter[0][1]},
                    float(coil_voltage) * .95, float(coil_voltage) * 1.05,
                    supply_voltage,
                )
            ):
                continue
            matched = False
            for output_pin, output_net in _pin(
                driver_ref, names, pad_nets, r"(?:OUT[1-7]|[1-7]C)",
            ):
                if output_net != coil_b:
                    continue
                ordinal = re.search(r"[1-7]", names.get((driver_ref, output_pin), ""))
                input_pattern = (
                    rf"(?:IN{ordinal.group(0)}|{ordinal.group(0)}B)"
                    if ordinal else r"$^"
                )
                input_net = next(
                    (
                        net for _, net in _pin(
                            driver_ref, names, pad_nets, input_pattern,
                        )
                    ),
                    None,
                )
                if (
                    input_net in pulled_down
                    and input_net is not None
                    and _physical_control_endpoint(
                        input_net, parts, pad_nets, {driver_ref, relay_ref}, powered_mcu_refs,
                    )
                ):
                    matched = True
                    break
            if matched:
                relay_complete += 1
                break
    if relay_complete:
        counts["relay_control_channel"] = relay_complete

    # TB6612: each dual H-bridge output pair must reach a real two-way motor
    # terminal.  The complete operating envelope is only published after both
    # supply rails are independently observed at reviewed-compatible voltages.
    tb_channels = 0
    tb_controls = True
    tb_support = True
    tb_functional_channels = 0
    for driver in [part for part in parts if _identity(part) == "tb6612fng"]:
        ref = str(driver.get("ref"))
        vm = _pin(ref, names, pad_nets, r"VM[123]")
        vcc = _pin(ref, names, pad_nets, r"VCC")
        pgnd = _pin(ref, names, pad_nets, r"(?:PGND[12]|GND)")
        stby = _pin(ref, names, pad_nets, r"STBY")
        driver_ground = {net for _, net in pgnd}
        if (
            len(vm) != 3 or len({net for _, net in vm}) != 1
            or len(vcc) != 1 or len(pgnd) != 5
            or len(driver_ground) != 1
        ):
            tb_support = False
            continue
        vm_net, vcc_net = vm[0][1], vcc[0][1]
        driver_pulled_down = _resistor_to_ground(parts, pad_nets, driver_ground)
        controls = _pin(ref, names, pad_nets, r"(?:AIN[12]|BIN[12]|PWM[AB])")
        controls_ok = (
            vm_net != vcc_net and vm_net not in ground and vcc_net not in ground
            and len(stby) == 1 and stby[0][1] in driver_pulled_down
            and len(controls) == 6
            and len({net for _, net in controls}) == 6
            and all(net in driver_pulled_down for _, net in controls)
        )
        if not controls_ok:
            tb_controls = False
        driver_channels = 0
        for pair_names in ((r"AO1", r"AO2"), (r"BO1", r"BO2")):
            one = _pin(ref, names, pad_nets, pair_names[0])
            two = _pin(ref, names, pad_nets, pair_names[1])
            if (
                len(one) != 2 or len(two) != 2
                or len({net for _, net in one}) != 1
                or len({net for _, net in two}) != 1
                or one[0][1] == two[0][1]
            ):
                continue
            if sum(
                {first, second} == {one[0][1], two[0][1]}
                for _, first, second in _terminal_pairs(parts, pad_nets)
            ) == 1:
                driver_channels += 1
        tb_channels += driver_channels
        vcc_caps = _caps_to_ground(parts, pad_nets, vcc_net, driver_ground)
        vm_caps = _caps_to_ground(parts, pad_nets, vm_net, driver_ground)
        support_ok = (
            {"10u", "100n"} <= {
                "10u" if "10u" in value else "100n" if "100n" in value else value
                for value in vcc_caps
            }
            and {"10u", "100n"} <= {
                "10u" if "10u" in value else "100n" if "100n" in value else value
                for value in vm_caps
            }
            and _tvs_clamps(parts, pad_nets, vm_net, driver_ground)
        )
        if not support_ok:
            tb_support = False
        active_controls = [stby[0][1]] + [net for _, net in controls] if stby else []
        functional_controls = (
            controls_ok
            and all(
                _physical_control_endpoint(net, parts, pad_nets, {ref}, powered_mcu_refs)
                for net in active_controls
            )
        )
        record = _record(driver)
        limits = record.operating_limits if record is not None else {}
        supply_ok = (
            _rail_voltage(
                observed_outputs, vm_net, driver_ground,
                float(limits.get("motor_voltage_min_v", 0)),
                float(limits.get("motor_voltage_max_v", -1)),
                supply_voltage,
            )
            and _rail_voltage(
                observed_outputs, vcc_net, driver_ground,
                float(limits.get("logic_voltage_min_v", 0)),
                float(limits.get("logic_voltage_max_v", -1)),
                supply_voltage,
            )
        )
        if driver_channels == 2 and functional_controls and support_ok and supply_ok:
            tb_functional_channels += driver_channels
    if tb_channels:
        counts["tb6612_motor_channel"] = tb_channels
        paths["tb6612_motor_channel_path"] = True
        if tb_controls:
            paths["tb6612_control_and_safe_default"] = True
        if tb_support:
            paths["tb6612_support_and_clamp"] = True
        metadata["tb6612_default_motor_current_a"] = .20
        metadata["tb6612_repeated_regeneration"] = "unverified"
    if tb_functional_channels:
        paths["tb6612_voltage_envelope"] = True

    # DRV8833 is a separate reviewed motor-controller capability.  It must
    # retain its own supply range, bypass and sleep-safe control; it never
    # establishes a TB6612 obligation.
    drv_channels = 0
    drv_safe = True
    drv_functional_channels = 0
    for driver in [part for part in parts if _identity(part) == "drv8833pwpr"]:
        ref = str(driver.get("ref"))
        vm = _pin(ref, names, pad_nets, r"VM")
        ground_pins = _pin(ref, names, pad_nets, r"GND")
        if len(vm) != 1 or not ground_pins or any(net not in ground for _, net in ground_pins):
            drv_safe = False
            continue
        driver_channels = 0
        for first_name, second_name in ((r"AOUT1", r"AOUT2"), (r"BOUT1", r"BOUT2")):
            first, second = _pin(ref, names, pad_nets, first_name), _pin(ref, names, pad_nets, second_name)
            if (
                len(first) == len(second) == 1
                and first[0][1] != second[0][1]
                and any(
                    {one, two} == {first[0][1], second[0][1]}
                    for _, one, two in _terminal_pairs(parts, pad_nets)
                )
            ):
                driver_channels += 1
        drv_channels += driver_channels
        inputs = _pin(ref, names, pad_nets, r"(?:AIN[12]|BIN[12]|SLEEP)")
        controls_ok = (
            len(inputs) == 5 and all(net in pulled_down for _, net in inputs)
            and all(
                _physical_control_endpoint(net, parts, pad_nets, {ref}, powered_mcu_refs)
                for _, net in inputs
            )
        )
        caps = _caps_to_ground(parts, pad_nets, vm[0][1], ground)
        record = _record(driver)
        limits = record.operating_limits if record is not None else {}
        support_ok = (
            any("100n" in value for value in caps)
            and _tvs_clamps(parts, pad_nets, vm[0][1], ground)
            and _rail_voltage(
                observed_outputs, vm[0][1], ground,
                float(limits.get("motor_supply_min_v", 0)),
                float(limits.get("motor_supply_max_v", -1)),
                supply_voltage,
            )
        )
        if not controls_ok or not support_ok:
            drv_safe = False
        if driver_channels == 2 and controls_ok and support_ok:
            drv_functional_channels += driver_channels
    if drv_channels:
        counts["drv8833_motor_channel"] = drv_channels
        paths["drv8833_motor_channel_path"] = True
        if drv_safe:
            paths["drv8833_control_and_clamp"] = True

    # A4988 requests are for the actual Pololu-compatible carrier socket, not
    # the bare IC.  Require every reviewed carrier contact, a board-side motor
    # boundary, deliberate inactive reset/sleep/enable states, control sources,
    # both returns, and the documented local VM bypass before publishing it.
    socket_count = 0
    socket_safe = True
    socket_functional = 0
    socket_screw_terminal_binding = 0
    for socket in [part for part in parts if _identity(part) == "a4988-carrier-socket"]:
        ref = str(socket.get("ref"))
        record = _record(socket)
        if record is None:
            continue
        pin = record.port_pins
        required = {
            "enable": pin["enable"], "ms1": pin["ms1"], "ms2": pin["ms2"],
            "ms3": pin["ms3"], "reset": pin["reset"], "sleep": pin["sleep"],
            "step": pin["step"], "dir": pin["dir"], "gnd_logic": pin["gnd_logic"],
            "vdd": pin["vdd"], "coil_1b": pin["coil_1b"], "coil_1a": pin["coil_1a"],
            "coil_2a": pin["coil_2a"], "coil_2b": pin["coil_2b"],
            "gnd_motor": pin["gnd_motor"], "vmot": pin["vmot"],
        }
        nets = {name: pad_nets.get((ref, number)) for name, number in required.items()}
        if (
            any(not net for net in nets.values())
            or nets["gnd_logic"] != nets["gnd_motor"]
            or len({nets["gnd_logic"], nets["vdd"], nets["vmot"]}) != 3
        ):
            continue
        coil_nets = {
            nets[name]
            for name in ("coil_1a", "coil_1b", "coil_2a", "coil_2b")
        }
        coil_pairs = {
            frozenset({nets["coil_1a"], nets["coil_1b"]}),
            frozenset({nets["coil_2a"], nets["coil_2b"]}),
        }
        connector_nets: list[tuple[set[str], set[str]]] = []
        for part in parts:
            terminal_record = _record(part)
            if terminal_record is None or str(part.get("ref")) == ref:
                continue
            features = terminal_record.physical_features
            if not {
                "screw-terminal", "jst-connector", "wire-to-board-connector",
            } & features:
                continue
            connector_nets.append((_part_nets(part, pad_nets), set(features)))
        endpoint_sets = {frozenset(nets) for nets, _ in connector_nets}
        motor_endpoint_bound = (
            frozenset(coil_nets) in endpoint_sets
            or coil_pairs <= endpoint_sets
        )
        screw_terminal_bound = any(
            nets == coil_nets and "screw-terminal" in features
            for nets, features in connector_nets
        )
        if len(coil_nets) != 4 or not motor_endpoint_bound:
            continue
        socket_count += 1
        defaults_ok = (
            _resistor_between(parts, pad_nets, nets["enable"], nets["vdd"])
            and all(
                net in pulled_down
                for net in (
                    nets["ms1"], nets["ms2"], nets["ms3"], nets["reset"], nets["sleep"],
                )
            )
        )
        caps = _caps_between(parts, pad_nets, nets["vmot"], nets["gnd_logic"])
        bypass_ok = any("100u" in value for value in caps) and any(
            "100n" in value for value in caps
        )
        supply_paths_ok = (
            _supply_return_endpoint(
                nets["vdd"], {nets["gnd_logic"]}, parts, pad_nets, {ref},
            )
            and _supply_return_endpoint(
                nets["vmot"], {nets["gnd_logic"]}, parts, pad_nets, {ref},
            )
        )
        controls_ok = all(
            _physical_control_endpoint(
                nets[name], parts, pad_nets, {ref}, powered_mcu_refs,
            )
            for name in ("enable", "reset", "sleep", "step", "dir")
        )
        if not (defaults_ok and bypass_ok and supply_paths_ok):
            socket_safe = False
        if defaults_ok and bypass_ok and supply_paths_ok and controls_ok:
            socket_functional += 1
        if defaults_ok and bypass_ok and supply_paths_ok and controls_ok and screw_terminal_bound:
            socket_screw_terminal_binding += 1
    if socket_count:
        counts["a4988_carrier_socket"] = socket_count
        paths["a4988_carrier_socket_representation"] = True
        if socket_safe:
            paths["a4988_carrier_socket_safe_default"] = True

    if socket_functional:
        gates["a4988_driver_socket"] = "pass"
    if socket_screw_terminal_binding:
        gates["a4988_socket_motor_terminal_binding"] = "pass"
    if relay_complete >= 2 and relay_contact_bindings >= 2:
        gates["external_relay_control_function"] = "pass"
        gates["external_relay_outputs_on_screw_terminals"] = "pass"
    # A carrier socket establishes an interface, not a fitted motor driver.
    if tb_functional_channels or drv_functional_channels:
        gates["external_motor_controller_function"] = "pass"
        gates["external_motor_screw_terminal_binding"] = "pass"
    if uln_functional or relay_complete or tb_functional_channels or drv_functional_channels:
        gates["external_actuator_driver_function"] = "pass"
    return {
        "net_paths": paths,
        "channel_counts": counts,
        "gates": gates,
        **metadata,
    }
