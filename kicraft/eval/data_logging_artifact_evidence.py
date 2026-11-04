"""Fail-closed hardware-capability evidence for nonvolatile data logging.

A positive result proves a delivered powered acquisition path into one complete,
powered MCU and a writable I2C EEPROM.  It does not claim generated firmware, a
logging schedule, EEPROM endurance, retention, or assembled behavior.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.validation import _capacitance_farads, _resistance_ohms
from kicraft.eval.application_pin_evidence import reviewed_application_pins

_STORAGE_IDENTITY = "24lc256t-i/sn"


def _record(part: Mapping[str, Any]):
    return physical_inventory_record(
        mpn=part.get("mpn"),
        symbol=part.get("symbol"),
        footprint=part.get("footprint"),
        datasheet=part.get("datasheet"),
        sourcing_note=part.get("sourcing_note"),
    )


def _net(
    pad_nets: Mapping[tuple[str, str], str], ref: str, pin: str,
) -> str | None:
    net = pad_nets.get((ref, pin))
    return net if isinstance(net, str) and net else None


def _passive_between(
    parts: list[dict[str, Any]],
    pad_nets: Mapping[tuple[str, str], str],
    first: str,
    second: str,
    *,
    prefix: str,
    minimum: float,
    maximum: float | None,
    parser: Callable[[str], float | None],
) -> bool:
    return any(
        str(part.get("ref") or "").upper().startswith(prefix)
        and {_net(pad_nets, str(part.get("ref") or ""), "1"), _net(pad_nets, str(part.get("ref") or ""), "2")}
        == {first, second}
        and (value := parser(str(part.get("value") or ""))) is not None
        and value >= minimum
        and (maximum is None or value <= maximum)
        for part in parts
    )


def _storage_wiring(
    parts: list[dict[str, Any]],
    pad_nets: Mapping[tuple[str, str], str],
    supply_voltage: Callable[[str, str], float | None],
) -> tuple[str, str, str, str, int] | None:
    """Return one complete writable EEPROM wiring tuple, or no positive evidence."""
    for part in parts:
        record = _record(part)
        if record is None or record.identity != _STORAGE_IDENTITY:
            continue
        ref = str(part.get("ref") or "")
        expected = {
            "vdd": "8", "ground": "4", "sda": "5", "scl": "6",
            "wp": "7", "a0": "1", "a1": "2", "a2": "3",
        }
        if any(str(record.port_pins.get(name) or "") != pin for name, pin in expected.items()):
            continue
        vdd, ground, sda, scl, wp = (
            _net(pad_nets, ref, expected[name])
            for name in ("vdd", "ground", "sda", "scl", "wp")
        )
        address_nets = [_net(pad_nets, ref, expected[name]) for name in ("a0", "a1", "a2")]
        if (
            not all((vdd, ground, sda, scl, wp))
            or len({vdd, ground, sda, scl}) != 4
            or wp != ground
            or any(net not in {ground, vdd} for net in address_nets)
        ):
            continue
        limits = record.operating_limits
        voltage = supply_voltage(vdd, ground)
        if not (
            isinstance(voltage, (int, float))
            and not isinstance(voltage, bool)
            and isinstance(limits.get("supply_min_v"), (int, float))
            and isinstance(limits.get("supply_max_v"), (int, float))
            and float(limits["supply_min_v"]) <= float(voltage) <= float(limits["supply_max_v"])
        ):
            continue
        if not _passive_between(
            parts, pad_nets, vdd, ground, prefix="C", minimum=100e-9,
            maximum=None, parser=_capacitance_farads,
        ):
            continue
        if not all(
            _passive_between(
                parts, pad_nets, vdd, bus, prefix="R", minimum=1_000.0,
                maximum=10_000.0, parser=_resistance_ohms,
            )
            for bus in (sda, scl)
        ):
            continue
        address = 0x50 + sum((1 << index) for index, net in enumerate(address_nets) if net == vdd)
        return vdd, ground, sda, scl, address
    return None


def i2c_adc_input_nets(
    parts: list[dict[str, Any]],
    pad_nets: Mapping[tuple[str, str], str],
    *,
    vdd: str,
    ground: str,
    sda: str | None,
    scl: str | None,
    supply_voltage: Callable[[str, str], float | None],
) -> set[str]:
    """ADS1115 inputs readable on a bus already bound to a qualified controller.

    Require the same supply domain, decoupling, bounded pull-ups and distinct
    valid address straps.  An isolated powered ADC is not a readable acquisition
    path.  Callers establish the complete powered controller owning this bus.
    """
    voltage = supply_voltage(vdd, ground)
    if (
        not sda or not scl or len({vdd, ground, sda, scl}) != 4
        or voltage is None or not 2.0 <= voltage <= 5.5
        or not all(
            _passive_between(
                parts, pad_nets, vdd, bus, prefix="R", minimum=1_000.0,
                maximum=10_000.0, parser=_resistance_ohms,
            )
            for bus in (sda, scl)
        )
    ):
        return set()
    addresses: set[str] = set()
    inputs: set[str] = set()
    for part in parts:
        record = _record(part)
        ref = str(part.get("ref") or "")
        if (
            record is None or record.identity != "ads1115idgsr"
            or (_net(pad_nets, ref, "9"), _net(pad_nets, ref, "10")) != (sda, scl)
        ):
            continue
        address = _net(pad_nets, ref, "1")
        if (
            (_net(pad_nets, ref, "8"), _net(pad_nets, ref, "3")) != (vdd, ground)
            or address not in {ground, vdd, sda, scl} or address in addresses
            or not _passive_between(
                parts, pad_nets, vdd, ground, prefix="C", minimum=100e-9,
                maximum=None, parser=_capacitance_farads,
            )
        ):
            return set()
        addresses.add(address)
        inputs.update(
            net for pin in ("4", "5", "6", "7")
            if (net := _net(pad_nets, ref, pin))
            and net not in {vdd, ground, sda, scl}
        )
    return inputs


def _logger_mcu(
    parts: list[dict[str, Any]],
    pad_nets: Mapping[tuple[str, str], str],
    *,
    sda: str,
    scl: str,
    vdd: str,
    ground: str,
    analog_nets: set[str],
    independent_input_nets: set[str],
    mcu_facts: Mapping[str, Any],
    powered_mcu_refs: set[str],
    supply_voltage: Callable[[str, str], float | None],
) -> str | None:
    """Require one complete powered MCU to own I2C and every acquisition path."""
    complete = {str(ref) for ref in mcu_facts.get("complete_mcu_refs", ())}
    i2c_sda = reviewed_application_pins("i2c-sda")
    i2c_scl = reviewed_application_pins("i2c-scl")
    adc = reviewed_application_pins("adc")
    inputs = reviewed_application_pins("input")
    for part in parts:
        ref = str(part.get("ref") or "")
        record = _record(part)
        if ref not in complete & powered_mcu_refs or record is None:
            continue
        supply = mcu_facts.get("complete_mcu_supplies", {}).get(ref)
        if not isinstance(supply, Mapping) or supply.get("rail") != vdd or supply.get("ground") != ground:
            continue
        if not any(_net(pad_nets, ref, pin) == sda for pin in i2c_sda.get(record.identity, set())):
            continue
        if not any(_net(pad_nets, ref, pin) == scl for pin in i2c_scl.get(record.identity, set())):
            continue
        adc_nets = {_net(pad_nets, ref, pin) for pin in adc.get(record.identity, set())}
        adc_nets.update(i2c_adc_input_nets(
            parts, pad_nets, vdd=vdd, ground=ground, sda=sda, scl=scl,
            supply_voltage=supply_voltage,
        ))
        input_nets = {_net(pad_nets, ref, pin) for pin in inputs.get(record.identity, set())}
        if analog_nets <= adc_nets and independent_input_nets <= adc_nets | input_nets:
            return ref
    return None


def _verified_nets(values: Iterable[object]) -> set[str]:
    return {value for value in values if isinstance(value, str) and value}


def data_logging_hardware_facts(
    parts: list[dict[str, Any]],
    names: Mapping[tuple[str, str], str],
    pad_nets: Mapping[tuple[str, str], str],
    *,
    mcu_facts: Mapping[str, Any],
    analog_facts: Mapping[str, Any],
    powered_mcu_refs: set[str],
    supply_voltage: Callable[[str, str], float | None],
    independent_input_nets: Iterable[object] = (),
) -> dict[str, Any]:
    """Publish ``external_data_logging_function`` as a hardware capability only.

    ``analog_output_nets`` is accepted only from the existing delivered-artifact
    analog observer, which publishes it only for a complete acquisition chain.
    ``independent_input_nets`` is an extension seam for another delivered-artifact
    observer; the caller must not populate it from labels, requirements, or prose.
    Firmware execution, scheduling, endurance and retention remain unobserved.
    """
    del names
    wiring = _storage_wiring(parts, pad_nets, supply_voltage)
    outputs = analog_facts.get("analog_output_nets") if isinstance(analog_facts, Mapping) else ()
    analog_nets = _verified_nets(
        outputs if isinstance(outputs, Iterable) and not isinstance(outputs, str) else ()
    )
    external_nets = _verified_nets(independent_input_nets)
    if wiring is None or not (analog_nets or external_nets):
        return {"net_paths": {}, "channel_counts": {}, "gates": {}}
    vdd, ground, sda, scl, address = wiring
    mcu_ref = _logger_mcu(
        parts, pad_nets, sda=sda, scl=scl, analog_nets=analog_nets,
        vdd=vdd, ground=ground,
        independent_input_nets=external_nets, mcu_facts=mcu_facts,
        powered_mcu_refs=powered_mcu_refs,
        supply_voltage=supply_voltage,
    )
    if mcu_ref is None:
        return {"net_paths": {}, "channel_counts": {}, "gates": {}}
    acquired = analog_nets | external_nets
    return {
        "net_paths": {
            "data_logging_i2c_eeprom": True,
            "data_logging_acquisition_paths": True,
        },
        "channel_counts": {
            "data_logging_storage": 1,
            "data_logging_acquisition": len(acquired),
        },
        "data_logging_storage_addresses": [f"0x{address:02X}"],
        "data_logging_mcu_refs": [mcu_ref],
        "gates": {
            "external_data_logging_function": "pass",
        },
    }
