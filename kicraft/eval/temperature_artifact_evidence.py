"""Physical TMP102 acquisition paths; inventory and MCU presence are insufficient."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.validation import _capacitance_farads, _resistance_ohms


def _passive_between(parts, pad_nets, symbol, left, right, parser, minimum, maximum):
    for part in parts:
        if part.get("symbol") not in {symbol, symbol + "_Small"}:
            continue
        ref = str(part.get("ref") or "")
        if {pad_nets.get((ref, "1")), pad_nets.get((ref, "2"))} != {left, right}:
            continue
        value = parser(str(part.get("value") or ""))
        if value is not None and minimum <= value <= maximum:
            return True
    return False


def temperature_acquisition_facts(
    parts: list[dict[str, Any]],
    pad_nets: Mapping[tuple[str, str], str],
    *,
    mcu_facts: Mapping[str, Any],
    powered_mcu_refs: set[str],
    supply_voltage: Callable[[str, str], float | None],
) -> dict[str, Any]:
    """Prove the reviewed 0x48 TMP102 circuit reaches a powered complete MCU.

    TMP102 converts autonomously after power-on. Application firmware must read
    register 0 over the connected bus; this observation is hardware, not a claim
    that acquisition firmware has been flashed or USB certification obtained.
    """
    acquisitions: set[str] = set()
    domains = mcu_facts.get("complete_mcu_supplies", {})
    gpio_nets = mcu_facts.get("complete_mcu_gpio_nets", {})
    for sensor in parts:
        record = physical_inventory_record(
            mpn=sensor.get("mpn"), symbol=sensor.get("symbol"),
            footprint=sensor.get("footprint"), datasheet=sensor.get("datasheet"),
            sourcing_note=sensor.get("sourcing_note"),
        )
        if record is None or record.identity != "tmp102aidrlr":
            continue
        ref = str(sensor.get("ref") or "")
        scl, ground, address, rail, sda = (
            pad_nets.get((ref, pin)) for pin in ("1", "2", "4", "5", "6")
        )
        if not (
            all(isinstance(net, str) and net for net in (scl, ground, rail, sda))
            and len({scl, ground, rail, sda}) == 4 and address == ground
        ):
            continue
        voltage = supply_voltage(rail, ground)
        if not (
            isinstance(voltage, (int, float)) and not isinstance(voltage, bool)
            and math.isfinite(voltage)
            and record.operating_limits["supply_min_v"] <= voltage <= record.operating_limits["supply_max_v"]
            and _passive_between(parts, pad_nets, "Device:C", rail, ground,
                                 _capacitance_farads, 100e-9, math.inf)
            and all(_passive_between(parts, pad_nets, "Device:R", bus, rail,
                                     _resistance_ohms, 1_000.0, 10_000.0)
                    for bus in (scl, sda))
        ):
            continue
        for mcu_ref in powered_mcu_refs:
            if domains.get(mcu_ref) != {"rail": rail, "ground": ground}:
                continue
            owned = [
                {source for source in gpio_nets.get(bus, ())
                 if isinstance(source, str) and source.rpartition(".")[0] == mcu_ref}
                for bus in (scl, sda)
            ]
            if any(len(pins) != 1 for pins in owned) or owned[0] == owned[1]:
                continue
            mcu = next((part for part in parts if part.get("ref") == mcu_ref), None)
            if mcu is None:
                continue
            mcu_record = physical_inventory_record(
                mpn=mcu.get("mpn"), symbol=mcu.get("symbol"),
                footprint=mcu.get("footprint"), datasheet=mcu.get("datasheet"),
                sourcing_note=mcu.get("sourcing_note"),
            )
            if mcu_record is None:
                continue
            usb_nets = {
                pad_nets.get((mcu_ref, mcu_record.port_pins[port]))
                for port in ("usb_dm", "usb_dp") if port in mcu_record.port_pins
            }
            if {scl, sda} & usb_nets:
                continue
            acquisitions.add(ref)
            break
    return {
        "gates": {"external_temperature_acquisition_function": "pass"} if acquisitions else {},
        "net_paths": {"i2c_temperature_acquisition": True} if acquisitions else {},
        "channel_counts": {"temperature_acquisition": len(acquisitions)} if acquisitions else {},
    }
