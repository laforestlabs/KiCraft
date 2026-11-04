"""Fail-closed delivered-board observations for display and indicator circuits.

This module consumes reconciled footprint rows and physical pad-to-net relations.  It
never accepts a recipe selection, reference designator, or net label as evidence.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from kicraft.design.part_identity import physical_inventory_record, reviewed_part
from kicraft.design.synthesis.validation import _capacitance_farads, _resistance_ohms
from kicraft.eval.application_pin_evidence import reviewed_application_pins


def _record(part: dict[str, Any]):
    return physical_inventory_record(
        mpn=part.get("mpn"), symbol=part.get("symbol"), footprint=part.get("footprint"),
        datasheet=part.get("datasheet"), sourcing_note=part.get("sourcing_note"),
    )


def _identity(part: dict[str, Any]) -> str:
    record = _record(part)
    return record.identity if record is not None else ""


def _nets(part: dict[str, Any], pad_nets: dict[tuple[str, str], str]) -> set[str]:
    ref = str(part.get("ref") or "")
    return {net for (owner, _), net in pad_nets.items() if owner == ref and net}


def _net(pad_nets: dict[tuple[str, str], str], ref: str, pin: str) -> str | None:
    return pad_nets.get((ref, str(pin)))


def _part_with_nets(parts, pad_nets, first: str, second: str, *, prefix: str, predicate) -> bool:
    return any(
        str(part.get("ref") or "").upper().startswith(prefix)
        and _nets(part, pad_nets) == {first, second}
        and predicate(part)
        for part in parts
    )


def _resistor_between(parts, pad_nets, first: str, second: str, *, minimum=0.0, maximum=float("inf")) -> bool:
    return _part_with_nets(
        parts, pad_nets, first, second, prefix="R",
        predicate=lambda part: (value := _resistance_ohms(str(part.get("value") or ""))) is not None
        and minimum <= value <= maximum,
    )


def _capacitor_refs_between(parts, pad_nets, first: str, second: str, *, minimum_f=100e-9, maximum_f=float("inf")) -> set[str]:
    return {
        str(part.get("ref") or "")
        for part in parts
        if str(part.get("ref") or "").upper().startswith("C")
        and _nets(part, pad_nets) == {first, second}
        and (value := _capacitance_farads(str(part.get("value") or ""))) is not None
        and minimum_f <= value <= maximum_f
    }


def _ground_nets(parts, names, pad_nets) -> set[str]:
    grounds = {
        net for endpoint, name in names.items()
        if str(name).upper() in {"GND", "VSS", "PGND", "AGND", "DGND", "0V"}
        and (net := pad_nets.get(endpoint))
    }
    # Delivered generic connectors lack semantic pin names; their return is
    # established by a reviewed IC's named ground contact, not by a net label.
    for part in parts:
        record = _record(part)
        if record is None:
            continue
        ref = str(part.get("ref") or "")
        for port in ("ground", "gnd"):
            if (pin := record.port_pins.get(port)) and (net := _net(pad_nets, ref, pin)):
                grounds.add(net)
    return grounds


def _connector_parts(parts) -> list[dict[str, Any]]:
    return [
        part for part in parts
        if (record := _record(part)) is not None
        and (
            "header" in record.physical_features
            or "connector" in record.family
            or record.family.endswith("-receptacle")
            or any("connector" in feature for feature in record.physical_features)
        )
    ]


def _powered_output_sources(
    parts, mcu_facts: dict[str, Any], supply_voltage,
) -> dict[str, set[tuple[str, float, float, float, str]]]:
    """Return only reviewed, supply-qualified per-pin drive capability.

    The published output levels are conditional on the part's own documented
    test supply, per-pin source load, and configuration.  Preserve that
    condition with the MCU reference rather than treating voltage limits as an
    unconditional board-level drive rating.
    """
    if not isinstance(mcu_facts, dict):
        return {}
    parts_by_ref = {str(part.get("ref") or ""): part for part in parts}
    supplies = mcu_facts.get("complete_mcu_supplies")
    gpio_nets = mcu_facts.get("complete_mcu_gpio_nets")
    if not isinstance(supplies, dict) or not isinstance(gpio_nets, dict):
        return {}
    reviewed = reviewed_application_pins("output")
    outputs: dict[str, set[tuple[str, float, float, float, str]]] = defaultdict(set)
    for net, endpoints in gpio_nets.items():
        if not isinstance(net, str) or not isinstance(endpoints, list):
            continue
        for endpoint in endpoints:
            if not isinstance(endpoint, str) or "." not in endpoint:
                continue
            ref, pin = endpoint.rsplit(".", 1)
            part = parts_by_ref.get(ref)
            record = _record(part) if part is not None else None
            supply = supplies.get(ref)
            conditions = (
                record.support_network.get("gpio_output_test_conditions")
                if record is not None else None
            )
            source_sink = (
                conditions.get("source_sink_current_a")
                if isinstance(conditions, dict) else None
            )
            source_current = (
                source_sink.get("source")
                if isinstance(source_sink, dict) else source_sink
            )
            if (
                record is None
                or pin not in reviewed.get(record.identity, set())
                or not isinstance(supply, dict)
                or not isinstance(conditions, dict)
                or not isinstance(source_current, (int, float))
                or source_current <= 0
                or not isinstance(conditions.get("source"), str)
                or not isinstance(conditions.get("table"), str)
            ):
                continue
            rail, ground = supply.get("rail"), supply.get("ground")
            if not isinstance(rail, str) or not isinstance(ground, str):
                continue
            voltage = supply_voltage(rail, ground)
            test_supply = conditions.get("supply_v")
            minimum = conditions.get("supply_min_exclusive_v")
            maximum = conditions.get("supply_max_exclusive_v")
            supply_qualified = test_supply == 3.3 or (
                isinstance(voltage, (int, float))
                and isinstance(minimum, (int, float))
                and isinstance(maximum, (int, float))
                and minimum < voltage < maximum
            )
            high = record.operating_limits.get("output_high_min_v_at_3v3")
            low = record.operating_limits.get("output_low_max_v_at_3v3")
            if (
                isinstance(voltage, (int, float)) and 3.25 <= voltage <= 3.35
                and supply_qualified
                and isinstance(high, (int, float)) and isinstance(low, (int, float))
            ):
                outputs[net].add(
                    (ground, float(high), float(low), float(source_current), ref)
                )
    return dict(outputs)


def _hct245_paths(parts, pad_nets, supply_voltage, sources) -> dict[str, tuple[str, str, str, str]]:
    """Map a fully safe 3.3-V-to-5-V HCT245 direction from A inputs to B outputs."""
    paths: dict[str, tuple[str, str, str, str]] = {}
    hct = reviewed_part("sn74hct245pwr-jsm")
    assert hct is not None
    input_high = hct.operating_limits["input_high_min_v"]
    input_low = hct.operating_limits["input_low_max_v"]
    for part in parts:
        if _identity(part) != "sn74hct245pwr-jsm":
            continue
        ref = str(part.get("ref") or "")
        vdd, ground, direction, enable = (_net(pad_nets, ref, pin) for pin in ("20", "10", "1", "19"))
        if not all((vdd, ground, direction, enable)) or len({vdd, ground, direction, enable}) != 2:
            continue
        if direction != vdd or enable != ground or not _capacitor_refs_between(parts, pad_nets, vdd, ground):
            continue
        voltage = supply_voltage(vdd, ground)
        if not isinstance(voltage, (int, float)) or not 4.5 <= voltage <= 5.5:
            continue
        active_sources: set[str] = set()
        valid = True
        hct_paths: dict[str, tuple[str, str, str, str]] = {}
        for a_pin, b_pin in zip(range(2, 10), range(18, 10, -1), strict=True):
            source, output = _net(pad_nets, ref, str(a_pin)), _net(pad_nets, ref, str(b_pin))
            if source is None:
                valid = False
                break
            if output in {vdd, ground}:
                valid = False
                break
            if source == ground:
                continue
            guarantees = sources.get(source, set())
            if (
                len(guarantees) != 1 or source in active_sources or not output
                or next(iter(guarantees))[0] != ground
                or next(iter(guarantees))[1] < input_high
                or next(iter(guarantees))[2] > input_low
            ):
                valid = False
                break
            active_sources.add(source)
            hct_paths[output] = (source, ref, vdd, ground)
        if valid:
            paths.update(hct_paths)
    return paths


def _translated_control(
    target: str, translations: dict[str, tuple[str, str, str, str]], sources, *, rail: str | None, ground: str, distinct: set[str],
) -> bool:
    row = translations.get(target)
    if row is None:
        return False
    source, _shifter, shifter_vdd, shifter_ground = row
    if (
        (rail is not None and shifter_vdd != rail)
        or shifter_ground != ground
        or source not in sources
        or source in distinct
    ):
        return False
    distinct.add(source)
    return True


def _ws2812_strip(parts, pad_nets, supply_voltage, sources) -> bool:
    translations = _hct245_paths(parts, pad_nets, supply_voltage, sources)
    for connector in _connector_parts(parts):
        ref = str(connector.get("ref") or "")
        rail, data, ground = (_net(pad_nets, ref, pin) for pin in ("1", "2", "3"))
        if not all((rail, data, ground)) or len({rail, data, ground}) != 3:
            continue
        voltage = supply_voltage(rail, ground)
        if not isinstance(voltage, (int, float)) or not 4.5 <= voltage <= 5.5:
            continue
        if not _capacitor_refs_between(parts, pad_nets, rail, ground):
            continue
        # The 330-ohm lead resistor must join one validated 5-V HCT output to
        # DIN; a part elsewhere called "series" does not qualify.
        for translated, (source, _shifter, shifter_vdd, shifter_ground) in translations.items():
            if source not in sources or shifter_vdd != rail or shifter_ground != ground:
                continue
            if _resistor_between(parts, pad_nets, translated, data, minimum=297.0, maximum=363.0):
                return True
    return False


def _four_digit_display(parts, pad_nets, supply_voltage, sources) -> bool:
    translations = _hct245_paths(parts, pad_nets, supply_voltage, sources)
    digit_pins = ("2", "11", "6", "7")
    segment_pins = ("14", "16", "20", "23", "21", "15", "17", "22")
    digit_segments = ("1", "2", "4", "5", "6", "7", "9", "10")
    for driver in (part for part in parts if _identity(part) == "max7219cwg"):
        ref = str(driver.get("ref") or "")
        vdd = _net(pad_nets, ref, "19")
        grounds = {_net(pad_nets, ref, pin) for pin in ("4", "9")}
        iset = _net(pad_nets, ref, "18")
        serial_inputs = [_net(pad_nets, ref, pin) for pin in ("1", "13", "12")]
        if not vdd or None in grounds or len(grounds) != 1 or not iset or not all(serial_inputs):
            continue
        ground = next(iter(grounds))
        voltage = supply_voltage(vdd, ground)
        if not isinstance(voltage, (int, float)) or not 4.0 <= voltage <= 5.5:
            continue
        small_caps = _capacitor_refs_between(parts, pad_nets, vdd, ground, minimum_f=100e-9, maximum_f=1e-6)
        bulk_caps = _capacitor_refs_between(parts, pad_nets, vdd, ground, minimum_f=1e-6)
        if not small_caps or not bulk_caps or small_caps == bulk_caps:
            continue
        if not _resistor_between(parts, pad_nets, vdd, iset, minimum=24_300.0, maximum=29_700.0):
            continue
        controls: set[str] = set()
        if not all(_translated_control(net, translations, sources, rail=vdd, ground=ground, distinct=controls) for net in serial_inputs):
            continue
        digit_nets = [_net(pad_nets, ref, pin) for pin in digit_pins]
        segment_nets = {_net(pad_nets, ref, pin) for pin in segment_pins}
        if None in digit_nets or len(set(digit_nets)) != 4 or None in segment_nets or len(segment_nets) != 8:
            continue
        attached = []
        for digit in (part for part in parts if _identity(part) == "cc56-12ewa"):
            dref = str(digit.get("ref") or "")
            cathodes = {_net(pad_nets, dref, pin) for pin in ("3", "8")}
            segments = {_net(pad_nets, dref, pin) for pin in digit_segments}
            if len(cathodes) == 1 and None not in cathodes and next(iter(cathodes)) in digit_nets and segments == segment_nets:
                attached.append(next(iter(cathodes)))
        if len(attached) == 4 and set(attached) == set(digit_nets):
            return True
    return False


def _matrix_interface(parts, pad_nets, supply_voltage, sources) -> bool:
    translations = _hct245_paths(parts, pad_nets, supply_voltage, sources)
    for connector in _connector_parts(parts):
        ref = str(connector.get("ref") or "")
        ground = {_net(pad_nets, ref, pin) for pin in ("4", "8", "16")}
        outputs = [_net(pad_nets, ref, pin) for pin in ("1", "2", "3", "5", "6", "7", "9", "10", "11", "12", "13", "14", "15")]
        if len(ground) != 1 or None in ground or not all(outputs) or len(set(outputs)) != 13:
            continue
        connector_ground = next(iter(ground))
        controls: set[str] = set()
        if all(_translated_control(net, translations, sources, rail=None, ground=connector_ground, distinct=controls) for net in outputs):
            return True
    return False


def _indicator_output_count(parts, names, pad_nets, sources, grounds) -> int:
    """Count separately MCU-driven, resistor-limited real LED circuits."""
    observed: set[str] = set()
    for led in parts:
        record = _record(led)
        if record is None or not (
            "led" in record.physical_features
            or "indicator-led" in record.physical_features
            or record.family.startswith("led-")
            or record.family.endswith("-led")
        ):
            continue
        ref = str(led.get("ref") or "")
        anode_pin = record.port_pins.get("anode", "A")
        cathode_pin = record.port_pins.get("cathode", "K")
        anode, cathode = _net(pad_nets, ref, anode_pin), _net(pad_nets, ref, cathode_pin)
        if not anode or not cathode or anode == cathode or cathode not in grounds:
            continue
        drivers = set()
        for output, guarantees in sources.items():
            if (
                len(guarantees) == 1
                and next(iter(guarantees))[0] == cathode
                and _resistor_between(parts, pad_nets, output, anode, minimum=50.0, maximum=100_000.0)
            ):
                drivers.add(output)
        if len(drivers) == 1:
            observed.update(drivers)
    return len(observed)


def _common_anode_rgb_indicator(parts, pad_nets, supply_voltage, sources) -> bool:
    """Prove all channels share one powered MCU, 5-V LED, and return."""
    for led in (part for part in parts if _identity(part) in {
        "aptf1616lseezgkqbkc", "xl-1615rgbc-rf",
    }):
        ref = str(led.get("ref") or "")
        record = _record(led)
        if record is None:
            continue
        anode = _net(pad_nets, ref, record.port_pins["common_anode"])
        if not anode:
            continue
        channels = (
            ("red_cathode", 1485.0, 1515.0),
            ("green_cathode", 811.0, 829.0),
            ("blue_cathode", 811.0, 829.0),
        )
        cathodes = [_net(pad_nets, ref, record.port_pins[name]) for name, _, _ in channels]
        if None in cathodes or len(set(cathodes)) != 3:
            continue
        for ground in {
            _net(pad_nets, str(driver.get("ref") or ""), "2")
            for driver in parts if _identity(driver) == "mmbt3904"
        } - {None}:
            voltage = supply_voltage(anode, ground)
            if not isinstance(voltage, (int, float)) or not 4.75 <= voltage <= 5.25:
                continue
            controls: set[str] = set()
            mcu_refs: set[str] = set()
            complete = True
            for cathode, (_name, minimum, maximum) in zip(cathodes, channels, strict=True):
                candidates = []
                for driver in (part for part in parts if _identity(part) == "mmbt3904"):
                    driver_ref = str(driver.get("ref") or "")
                    base = _net(pad_nets, driver_ref, "1")
                    emitter = _net(pad_nets, driver_ref, "2")
                    collector = _net(pad_nets, driver_ref, "3")
                    if (
                        not base or not collector or emitter != ground
                        or not _resistor_between(
                            parts, pad_nets, cathode, collector,
                            minimum=minimum, maximum=maximum,
                        )
                        or not _resistor_between(
                            parts, pad_nets, base, ground,
                            minimum=99_000.0, maximum=101_000.0,
                        )
                    ):
                        continue
                    control = next(
                        (
                            output for output, guarantees in sources.items()
                            if len(guarantees) == 1
                            and (guarantee := next(iter(guarantees)))[0] == ground
                            and guarantee[1] > guarantee[2]
                            and guarantee[3] >= guarantee[1] / 4_700.0
                            and guarantee[3] > 0
                            and _resistor_between(
                                parts, pad_nets, output, base,
                                minimum=4_653.0, maximum=4_747.0,
                            )
                        ),
                        None,
                    )
                    if control is not None:
                        candidates.append((control, next(iter(sources[control]))[4]))
                if len(candidates) != 1:
                    complete = False
                    break
                control, mcu_ref = candidates[0]
                if control in controls:
                    complete = False
                    break
                controls.add(control)
                mcu_refs.add(mcu_ref)
            if complete and len(controls) == 3 and len(mcu_refs) == 1:
                return True
    return False


def _power_led_driver(parts, names, pad_nets, supply_voltage, sources) -> bool:
    # Reuse the established AL8860 loop proof for current setting, catch diode,
    # inductor, emitter/terminal, and input bypass. Add actual supply and CTRL
    # proof because a current-loop fact alone does not prove a usable driver.
    from .electrical_artifact_evidence import _reviewed_led_current_loop

    for driver in (part for part in parts if _identity(part) == "al8860mp-13"):
        ref = str(driver.get("ref") or "")
        current = _reviewed_led_current_loop(
            parts, names, pad_nets, controller_ref=ref
        )
        if not isinstance(current, (int, float)) or current < 0.95:
            continue
        vin, ground, control = (_net(pad_nets, ref, pin) for pin in ("8", "2", "4"))
        if not vin or not ground or not control or vin == ground:
            continue
        voltage = supply_voltage(vin, ground)
        if not isinstance(voltage, (int, float)) or not 4.5 <= voltage <= 40.0:
            continue
        limits = _record(driver).operating_limits
        minimum = limits["control_full_current_min_v"]
        maximum = limits["control_drive_max_v"]
        control_voltage = supply_voltage(control, ground)
        if isinstance(control_voltage, (int, float)) and minimum <= control_voltage <= maximum:
            return True
        if (
            len(sources.get(control, set())) == 1
            and next(iter(sources[control]))[0] == ground
            and next(iter(sources[control]))[1] >= minimum
        ):
            return True
    return False


def display_circuit_facts(
    parts: list[dict[str, Any]], names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str], *, mcu_facts: dict[str, Any],
    supply_voltage: Callable[[str, str], float | None],
) -> dict[str, Any]:
    """Return only display/indicator gates proven by delivered circuit topology."""
    sources = _powered_output_sources(parts, mcu_facts, supply_voltage)
    grounds = _ground_nets(parts, names, pad_nets)
    gates: dict[str, str] = {}
    counts: dict[str, int] = {}
    paths: dict[str, bool] = {}

    ws2812 = _ws2812_strip(parts, pad_nets, supply_voltage, sources)
    four_digit = _four_digit_display(parts, pad_nets, supply_voltage, sources)
    matrix = _matrix_interface(parts, pad_nets, supply_voltage, sources)
    rgb_indicator = _common_anode_rgb_indicator(parts, pad_nets, supply_voltage, sources)
    power_led = _power_led_driver(parts, names, pad_nets, supply_voltage, sources)
    indicators = _indicator_output_count(parts, names, pad_nets, sources, grounds)

    if ws2812:
        gates["external_ws2812_strip_interface"] = "pass"
        paths["ws2812_external_strip_control"] = True
    if four_digit:
        gates["external_four_digit_seven_segment_display"] = "pass"
        paths["four_digit_seven_segment_control"] = True
    if matrix:
        gates["external_led_matrix_function"] = "pass"
        paths["led_matrix_hub75_control"] = True
    if power_led:
        gates["external_led_driver_function"] = "pass"
        gates["external_three_watt_power_led_function"] = "pass"
        paths["three_watt_power_led_driver"] = True
    if rgb_indicator:
        counts["rgb_status_led_channel"] = 3
        gates["external_rgb_status_led_channels"] = "pass"
        paths["rgb_status_led_channel_control"] = True
    if indicators >= 4:
        counts["independent_indicator_output"] = indicators
        gates["external_four_indicator_outputs"] = "pass"
        gates["external_indicator_controller_function"] = "pass"
        paths["four_indicator_control"] = True
    return {"net_paths": paths, "channel_counts": counts, "gates": gates}
