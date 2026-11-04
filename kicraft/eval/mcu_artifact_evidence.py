"""Fail-closed MCU-core and conditional-expander evidence from delivered boards."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.validation import _capacitance_farads, _resistance_ohms
from kicraft.eval.application_pin_evidence import reviewed_application_pins


def _record(part: dict[str, Any]):
    return physical_inventory_record(
        mpn=part.get("mpn"), symbol=part.get("symbol"), footprint=part.get("footprint"),
        datasheet=part.get("datasheet"), sourcing_note=part.get("sourcing_note"),
    )


def _is_identity(part: dict[str, Any], identity: str) -> bool:
    record = _record(part)
    return record is not None and record.identity == identity


def _net(pad_nets: dict[tuple[str, str], str], ref: str, pin: str) -> str | None:
    return pad_nets.get((ref, pin))


def _passive_between(parts, pad_nets, first: str, second: str, *, prefix: str, minimum: float, parser) -> bool:
    return any(
        str(part.get("ref") or "").startswith(prefix)
        and {_net(pad_nets, str(part.get("ref")), "1"), _net(pad_nets, str(part.get("ref")), "2")} == {first, second}
        and (parsed := parser(str(part.get("value") or ""))) is not None and parsed >= minimum
        for part in parts
    )


def _cap_between(parts, pad_nets, first: str, second: str, minimum_f: float = 100e-9) -> bool:
    return _passive_between(parts, pad_nets, first, second, prefix="C", minimum=minimum_f, parser=_capacitance_farads)


def _cap_count_between(parts, pad_nets, first: str, second: str, minimum_f: float = 100e-9) -> int:
    return sum(
        str(part.get("ref") or "").startswith("C")
        and {_net(pad_nets, str(part.get("ref")), "1"), _net(pad_nets, str(part.get("ref")), "2")}
        == {first, second}
        and (parsed := _capacitance_farads(str(part.get("value") or ""))) is not None
        and parsed >= minimum_f
        for part in parts
    )


def _res_between(parts, pad_nets, first: str, second: str, minimum_ohm: float = 1_000.0) -> bool:
    return _passive_between(parts, pad_nets, first, second, prefix="R", minimum=minimum_ohm, parser=_resistance_ohms)


def _link_between(parts, pad_nets, first: str, second: str, prefix: str) -> bool:
    return any(
        str(part.get("ref") or "").startswith(prefix)
        and {_net(pad_nets, str(part.get("ref")), "1"), _net(pad_nets, str(part.get("ref")), "2")} == {first, second}
        for part in parts
    )


def _i2c_reaches_supported_mcu(parts, pad_nets, sda: str | None, scl: str | None) -> bool:
    for part in parts:
        ref = str(part.get("ref") or "")
        if _is_identity(part, "stm32g030f6p6"):
            if (_net(pad_nets, ref, "2"), _net(pad_nets, ref, "1")) == (sda, scl):
                return True
        elif _is_identity(part, "attiny1604-ssn"):
            if (_net(pad_nets, ref, "12"), _net(pad_nets, ref, "13")) == (sda, scl):
                return True
    return False


def _external_endpoints(by_net: dict[str, set[str]], net: str, owners: set[str]) -> bool:
    return bool(net and by_net[net] - owners) and not net.upper().startswith(("GND", "VDD", "+3V3", "+5V"))


def _one_physical_header_reaches(
    pad_nets: dict[tuple[str, str], str],
    nets: set[str],
) -> bool:
    """Require one connector footprint, not five independently exposed test pads."""
    header_nets: dict[str, set[str]] = defaultdict(set)
    for (ref, _pin), net in pad_nets.items():
        if net and ref.upper().startswith(("J", "P", "CN", "CONN", "H")):
            header_nets[ref].add(net)
    return any(nets <= exposed for exposed in header_nets.values())


def _usb_data_connector_reaches(
    parts, pad_nets: dict[tuple[str, str], str], usb_dm: str, usb_dp: str,
) -> bool:
    """Prove separate data paths to the same delivered connector footprint."""
    from kicraft.eval.electrical_artifact_evidence import _reaches, _usb_data_graph

    if not usb_dm or not usb_dp or usb_dm == usb_dp:
        return False
    graph = _usb_data_graph(parts, pad_nets)
    if _reaches(graph, usb_dm, usb_dp):
        return False
    connectors: dict[str, dict[str, str]] = defaultdict(dict)
    for (ref, pin), net in pad_nets.items():
        if net and ref.upper().startswith(("J", "P", "CN", "CONN", "H")):
            connectors[ref][pin] = net
    parts_by_ref = {str(part.get("ref") or ""): part for part in parts}
    for ref, contacts in connectors.items():
        record = _record(parts_by_ref.get(ref, {}))
        if record is not None and "usb-c-receptacle" in record.physical_features:
            dm_nets = {contacts[pin] for pin in ("A7", "B7") if pin in contacts}
            dp_nets = {contacts[pin] for pin in ("A6", "B6") if pin in contacts}
            if (
                any(_reaches(graph, usb_dm, net) for net in dp_nets)
                or any(_reaches(graph, usb_dp, net) for net in dm_nets)
            ):
                continue
        else:
            dm_nets = dp_nets = set(contacts.values())
        if any(
            dm != dp and _reaches(graph, usb_dm, dm) and _reaches(graph, usb_dp, dp)
            and not _reaches(graph, usb_dm, dp) and not _reaches(graph, usb_dp, dm)
            for dm in dm_nets for dp in dp_nets
        ):
            return True
    return False

def _rom_uart_header_reaches(parts, pad_nets, ref, uart_pins, vdd, gnd, en, boot):
    """Prove the reviewed six contacts, not merely a named programming part."""
    tx, rx = (_net(pad_nets, ref, pin) for pin in uart_pins)
    expected = (vdd, gnd, tx, rx, en, boot)
    if not all(expected) or len(set(expected)) != 6:
        return False
    return any(
        str(part.get("ref") or "").upper().startswith(("J", "P", "CN"))
        and str(part.get("mpn") or "").upper() == "68000-406HLF"
        and all(
            _net(pad_nets, str(part.get("ref")), str(index)) == net
            for index, net in enumerate(expected, start=1)
        )
        for part in parts
    )




def _module_core_complete(
    parts,
    pad_nets: dict[tuple[str, str], str],
    ref: str,
    *,
    vdd_pin: str,
    ground_pins: tuple[str, ...],
    enable_pin: str,
    boot_pin: str,
    usb_pins: tuple[str, str],
    uart_pins: tuple[str, str] | None = None,
    boot_high_pin: str | None = None,
) -> bool:
    """Validate the reviewed Espressif recovery circuit on delivered pads."""
    vdd = _net(pad_nets, ref, vdd_pin)
    grounds = {net for pin in ground_pins if (net := _net(pad_nets, ref, pin))}
    gnd = next(iter(grounds), None)
    en, boot = (_net(pad_nets, ref, pin) for pin in (enable_pin, boot_pin))
    usb_dm, usb_dp = (_net(pad_nets, ref, pin) for pin in usb_pins)
    boot_high = _net(pad_nets, ref, boot_high_pin) if boot_high_pin else vdd
    return bool(
        vdd and gnd and len(grounds) == 1 and all(
            _net(pad_nets, ref, pin) == gnd for pin in ground_pins
        ) and vdd != gnd
        and en and boot
        and _cap_count_between(parts, pad_nets, vdd, gnd) >= 2
        and _cap_between(parts, pad_nets, vdd, gnd, 1e-6)
        and _res_between(parts, pad_nets, vdd, en, 10_000.0)
        and _cap_between(parts, pad_nets, en, gnd, 1e-6)
        and _link_between(parts, pad_nets, en, gnd, "SW")
        and _res_between(parts, pad_nets, vdd, boot, 10_000.0)
        and _link_between(parts, pad_nets, boot, gnd, "SW")
        and (boot_high_pin is None or _res_between(parts, pad_nets, vdd, boot_high, 10_000.0))
        and (
            _usb_data_connector_reaches(parts, pad_nets, usb_dm, usb_dp)
            or (uart_pins is not None and _rom_uart_header_reaches(
                parts, pad_nets, ref, uart_pins, vdd, gnd, en, boot,
            ))
        )
    )


def _complete_gpio_paths(
    complete: dict[str, set[str]],
    pad_nets: dict[tuple[str, str], str],
) -> tuple[int, dict[str, list[str]]]:
    """Count unique, externally exposed GPIO nets of completed MCU cores."""
    by_net: dict[str, set[str]] = defaultdict(set)
    for (ref, _pin), net in pad_nets.items():
        if net:
            by_net[net].add(ref)
    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ref, pins in complete.items():
        for pin in pins:
            net = _net(pad_nets, ref, pin)
            if (
                net
                and not net.upper().startswith(("GND", "VDD", "+3V3", "+5V"))
                and any(other != ref and other.upper().startswith(("J", "P", "CN", "CONN", "H"))
                        for other in by_net[net])
            ):
                candidates[net].append((ref, pin))
    # A shorted pair of GPIO pads is not two independently usable outputs.
    paths = {
        net: [f"{ref}.{pin}" for ref, pin in endpoints]
        for net, endpoints in candidates.items()
        if len(endpoints) == 1
    }
    return len(paths), paths


def _complete_gpio_nets(
    complete: dict[str, set[str]],
    pad_nets: dict[tuple[str, str], str],
) -> dict[str, list[str]]:
    """Map only unshorted application GPIO pads of completed cores."""
    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ref, pins in complete.items():
        for pin in pins:
            net = _net(pad_nets, ref, pin)
            if net and not net.upper().startswith(("GND", "VDD", "+3V3", "+5V")):
                candidates[net].append((ref, pin))
    return {
        net: [f"{ref}.{pin}" for ref, pin in endpoints]
        for net, endpoints in candidates.items()
        if len(endpoints) == 1
    }


def _rp2040_bootsel_path(
    parts,
    pad_nets: dict[tuple[str, str], str],
    qspi_cs: str | None,
    gnd: str | None,
) -> bool:
    if not qspi_cs or not gnd:
        return False
    switch_nets = [
        {
            net
            for net in (
                _net(pad_nets, str(part.get("ref")), "1"),
                _net(pad_nets, str(part.get("ref")), "2"),
            )
            if net
        }
        for part in parts
        if str(part.get("ref") or "").upper().startswith(("SW", "S", "JP"))
    ]
    return any(
        gnd in switch
        and any(
            str(part.get("ref") or "").upper().startswith("R")
            and qspi_cs in {
                _net(pad_nets, str(part.get("ref")), "1"),
                _net(pad_nets, str(part.get("ref")), "2"),
            }
            and bool(
                {
                    _net(pad_nets, str(part.get("ref")), "1"),
                    _net(pad_nets, str(part.get("ref")), "2"),
                }
                & switch
            )
            and (
                (parsed := _resistance_ohms(str(part.get("value") or ""))) is not None
                and parsed >= 1_000.0
            )
            for part in parts
        )
        for switch in switch_nets
    )


def mcu_circuit_facts(parts, names, pad_nets) -> dict[str, Any]:
    """Publish only exact reviewed-package and pad-graph MCU evidence."""
    paths: dict[str, bool] = {}
    counts: dict[str, int] = {}
    complete_gpio_pins: dict[str, set[str]] = {}
    complete_refs: set[str] = set()
    complete_mcu_supplies: dict[str, dict[str, str]] = {}
    by_net: dict[str, set[str]] = defaultdict(set)
    for (ref, _pin), net in pad_nets.items():
        if net:
            by_net[net].add(ref)
    for part in parts:
        ref = str(part.get("ref") or "")
        if _is_identity(part, "stm32g030f6p6"):
            vdd, gnd, nrst, swdio, swclk = (_net(pad_nets, ref, pin) for pin in ("4", "5", "6", "18", "19"))
            complete = bool(
                vdd and gnd and nrst and swdio and swclk and vdd != gnd
                and _cap_between(parts, pad_nets, vdd, gnd)
                and _res_between(parts, pad_nets, nrst, vdd, 10_000.0)
                and _cap_between(parts, pad_nets, nrst, gnd)
                and _link_between(parts, pad_nets, nrst, gnd, "SW")
                and _res_between(parts, pad_nets, swclk, gnd, 10_000.0)
                and _one_physical_header_reaches(pad_nets, {vdd, gnd, nrst, swdio, swclk})
            )
            if complete:
                paths["stm32g030f6p6_complete_core"] = True
                complete_refs.add(ref)
                complete_gpio_pins[ref] = {
                    str(pin) for pin in range(1, 21) if str(pin) not in {"4", "5", "6", "18", "19"}
                }
                complete_mcu_supplies[ref] = {"rail": vdd, "ground": gnd}
                counts["stm32g030f6p6_core"] = counts.get("stm32g030f6p6_core", 0) + 1
        elif _is_identity(part, "attiny1604-ssn"):
            vdd, gnd, updi = (_net(pad_nets, ref, pin) for pin in ("1", "14", "10"))
            complete = bool(
                vdd and gnd and updi and vdd != gnd
                and _cap_between(parts, pad_nets, vdd, gnd)
                and _res_between(parts, pad_nets, updi, vdd, 10_000.0)
                and _one_physical_header_reaches(pad_nets, {vdd, gnd, updi})
            )
            if complete:
                paths["attiny1604_ssn_complete_core"] = True
                counts["attiny1604_core"] = counts.get("attiny1604_core", 0) + 1
                complete_refs.add(ref)
                complete_gpio_pins[ref] = {str(pin) for pin in (2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13)}
                complete_mcu_supplies[ref] = {"rail": vdd, "ground": gnd}
        elif _is_identity(part, "rp2040"):
            vdd, gnd, testen, vreg_in, vreg_out, run, swclk, swdio = (
                _net(pad_nets, ref, pin)
                for pin in ("1", "57", "19", "44", "45", "26", "24", "25")
            )
            qspi = {
                name: _net(pad_nets, ref, pin)
                for name, pin in {
                    "cs": "56",
                    "sclk": "52",
                    "sd0": "53",
                    "sd1": "55",
                    "sd2": "54",
                    "sd3": "51",
                }.items()
            }
            flash = next((row for row in parts if _is_identity(row, "w25q16jvss")), None)
            flash_qspi = bool(
                flash
                and all(
                    _net(pad_nets, str(flash.get("ref")), pin) == qspi[name]
                    for name, pin in {
                        "cs": "1",
                        "sclk": "6",
                        "sd0": "5",
                        "sd1": "2",
                        "sd2": "3",
                        "sd3": "7",
                    }.items()
                )
                and _net(pad_nets, str(flash.get("ref")), "4") == gnd
                and _net(pad_nets, str(flash.get("ref")), "8") == vdd
            )
            io_decoupling = sum(
                str(row.get("ref") or "").upper().startswith("C")
                and {
                    _net(pad_nets, str(row.get("ref")), "1"),
                    _net(pad_nets, str(row.get("ref")), "2"),
                }
                == {vdd, gnd}
                and (
                    (parsed := _capacitance_farads(str(row.get("value") or ""))) is not None
                    and parsed >= 100e-9
                )
                for row in parts
            )
            complete = bool(
                vdd
                and gnd
                and testen == gnd
                and vreg_in == vdd
                and vreg_out not in {None, gnd, vdd}
                and run
                and swclk
                and swdio
                and io_decoupling >= 8
                and _cap_between(parts, pad_nets, vreg_out, gnd, 1e-6)
                and _res_between(parts, pad_nets, run, vdd, 10_000.0)
                and _link_between(parts, pad_nets, run, gnd, "SW")
                and _rp2040_bootsel_path(parts, pad_nets, qspi["cs"], gnd)
                and flash_qspi
                and _one_physical_header_reaches(
                    pad_nets, {vdd, gnd, run, swclk, swdio}
                )
            )
            if complete:
                paths["rp2040_swd_recovery_complete_core"] = True
                counts["rp2040_core"] = counts.get("rp2040_core", 0) + 1
                complete_refs.add(ref)
                complete_gpio_pins[ref] = {
                    "2", "3", "4", "5", "6", "7", "8", "9", "11", "12", "13", "14",
                    "15", "16", "17", "18", "27", "28", "29", "30", "31", "32", "34",
                    "35", "36", "37", "38", "39", "40", "41",
                }
                complete_mcu_supplies[ref] = {"rail": vdd, "ground": gnd}
        elif any(_is_identity(part, identity) for identity in (
            "esp32-c3-mini-1-n4", "esp32-c3-mini-1-h4x",
        )):
            complete = _module_core_complete(
                parts, pad_nets, ref, vdd_pin="3",
                ground_pins=("1", "2", "11", "14", *tuple(str(pin) for pin in range(36, 54))),
                enable_pin="8", boot_pin="23", usb_pins=("26", "27"), boot_high_pin="22",
                uart_pins=("31", "30") if _is_identity(part, "esp32-c3-mini-1-h4x") else None,
            )
            if complete:
                paths["esp32_c3_complete_core"] = True
                counts["esp32_c3_core"] = counts.get("esp32_c3_core", 0) + 1
                complete_refs.add(ref)
                complete_gpio_pins[ref] = {"5", "6", "12", "13", "16", "18", "19", "20", "21"}
                complete_mcu_supplies[ref] = {
                    "rail": _net(pad_nets, ref, "3"), "ground": _net(pad_nets, ref, "1"),
                }
        elif _is_identity(part, "esp32-s3-mini-1u-n4r2"):
            complete = _module_core_complete(
                parts, pad_nets, ref, vdd_pin="3",
                ground_pins=(
                    "1", "2", "42", "43", *tuple(str(pin) for pin in range(46, 66)),
                ),
                enable_pin="45", boot_pin="4", usb_pins=("23", "24"),
                uart_pins=("39", "40"),
            )
            # GPIO45 selects 3.3 V VDD_SPI through its internal pull-down;
            # IO26 is connected to the N4R2's embedded PSRAM.
            complete = complete and not _net(pad_nets, ref, "41") and not _net(pad_nets, ref, "26")
            if complete:
                paths["esp32_s3_complete_core"] = True
                counts["esp32_s3_core"] = counts.get("esp32_s3_core", 0) + 1
                complete_refs.add(ref)
                complete_gpio_pins[ref] = {
                    "5", "6", *tuple(str(pin) for pin in range(8, 23)),
                    "25", *tuple(str(pin) for pin in range(27, 39)),
                }
                complete_mcu_supplies[ref] = {
                    "rail": _net(pad_nets, ref, "3"), "ground": _net(pad_nets, ref, "1"),
                }
        elif _is_identity(part, "esp32-s3-mini-1-n8"):
            complete = _module_core_complete(
                parts, pad_nets, ref, vdd_pin="3",
                ground_pins=(
                    "1", "2", "42", "43", *tuple(str(pin) for pin in range(46, 61)), "GND",
                ),
                enable_pin="45", boot_pin="4", usb_pins=("23", "24"),
            )
            if complete:
                paths["esp32_s3_complete_core"] = True
                counts["esp32_s3_core"] = counts.get("esp32_s3_core", 0) + 1
                complete_refs.add(ref)
                complete_gpio_pins[ref] = {
                    "5", "6", *tuple(str(pin) for pin in range(8, 23)),
                    *tuple(str(pin) for pin in range(25, 39)),
                }
                complete_mcu_supplies[ref] = {
                    "rail": _net(pad_nets, ref, "3"), "ground": _net(pad_nets, ref, "1"),
                }
        elif _is_identity(part, "esp32-s3-wroom-1-n8r8") or _is_identity(part, "esp32-s3-wroom-1-n16r8"):
            complete = _module_core_complete(
                parts, pad_nets, ref, vdd_pin="2", ground_pins=("1", "40", "41"),
                enable_pin="3", boot_pin="27", usb_pins=("13", "14"),
            )
            if complete:
                paths["esp32_s3_complete_core"] = True
                counts["esp32_s3_core"] = counts.get("esp32_s3_core", 0) + 1
                complete_refs.add(ref)
                complete_gpio_pins[ref] = {
                    "4", "5", "6", "7", "8", "9", "10", "11", "12", "15", "16", "17",
                    "18", "19", "20", "21", "22", "23", "24", "25", "26", "31", "32",
                    "33", "34", "35", "38", "39",
                }
                complete_mcu_supplies[ref] = {
                    "rail": _net(pad_nets, ref, "2"), "ground": _net(pad_nets, ref, "1"),
                }
        elif any(_is_identity(part, identity) for identity in (
            "ch32v003j4m6", "ch32v003f4p6", "ch32v003f4u6",
        )):
            record = _record(part)
            vdd, gnd, swio = (
                _net(pad_nets, ref, record.port_pins[port]) for port in ("vdd", "vss", "swio")
            )
            complete = bool(
                vdd and gnd and swio and vdd != gnd
                and _cap_between(parts, pad_nets, vdd, gnd)
                and _one_physical_header_reaches(pad_nets, {vdd, gnd, swio})
            )
            if "vss_ep" in record.port_pins:
                complete = complete and _net(pad_nets, ref, record.port_pins["vss_ep"]) == gnd
            if "nrst" in record.port_pins:
                reset = _net(pad_nets, ref, record.port_pins["nrst"])
                complete = bool(
                    complete and reset and len({vdd, gnd, swio, reset}) == 4
                    and _res_between(parts, pad_nets, vdd, reset, 10_000.0)
                    and _link_between(parts, pad_nets, reset, gnd, "SW")
                    and _one_physical_header_reaches(pad_nets, {vdd, gnd, swio, reset})
                )
            if complete:
                paths[f"{record.identity}_complete_core"] = True
                counts["ch32v003_core"] = counts.get("ch32v003_core", 0) + 1
                complete_refs.add(ref)
                complete_gpio_pins[ref] = reviewed_application_pins("gpio").get(record.identity, set())
                complete_mcu_supplies[ref] = {"rail": vdd, "ground": gnd}
    ads = [part for part in parts if _is_identity(part, "ads1115idgsr")]
    if len(ads) == 2:
        refs = {str(part.get("ref")) for part in ads}
        supplies = [(_net(pad_nets, str(part.get("ref")), "8"), _net(pad_nets, str(part.get("ref")), "3")) for part in ads]
        buses = [(_net(pad_nets, str(part.get("ref")), "9"), _net(pad_nets, str(part.get("ref")), "10")) for part in ads]
        addresses = [_net(pad_nets, str(part.get("ref")), "1") for part in ads]
        inputs = [_net(pad_nets, str(part.get("ref")), pin) for part in ads for pin in ("4", "5", "6", "7")]
        powered = all(vdd and gnd and vdd != gnd and _cap_between(parts, pad_nets, vdd, gnd) for vdd, gnd in supplies)
        same_bus = len(set(buses)) == 1 and buses[0][0] and buses[0][1] and buses[0][0] != buses[0][1]
        pullups = same_bus and all(_res_between(parts, pad_nets, supplies[0][0], net, 4_700.0) for net in buses[0])
        address_valid = len(set(addresses)) == 2 and all(address in {supplies[index][0], supplies[index][1]} for index, address in enumerate(addresses))
        endpoints = all(net and _external_endpoints(by_net, net, refs) for net in inputs)
        if powered and same_bus and pullups and address_valid and _i2c_reaches_supported_mcu(parts, pad_nets, *buses[0]) and len(set(inputs)) == 8 and endpoints:
            paths["ads1115x2_eight_channel_adc"] = True
            counts["adc_input"] = 8
    mcp = [part for part in parts if _is_identity(part, "mcp23017-e/so")]
    if len(mcp) == 1:
        part = mcp[0]; ref = str(part.get("ref")); vdd, gnd = _net(pad_nets, ref, "9"), _net(pad_nets, ref, "10")
        sda, scl, reset = (_net(pad_nets, ref, pin) for pin in ("13", "12", "18"))
        outputs = [_net(pad_nets, ref, str(pin)) for pin in (*range(21, 29), *range(1, 9))]
        complete = bool(
            vdd and gnd and sda and scl and reset and vdd != gnd and sda != scl
            and _cap_between(parts, pad_nets, vdd, gnd)
            and _res_between(parts, pad_nets, reset, vdd, 10_000.0)
            and all(_net(pad_nets, ref, pin) == gnd for pin in ("15", "16", "17"))
            and _res_between(parts, pad_nets, vdd, sda, 4_700.0) and _res_between(parts, pad_nets, vdd, scl, 4_700.0)
            and _i2c_reaches_supported_mcu(parts, pad_nets, sda, scl)
            and len(set(outputs)) == 16 and all(net and _external_endpoints(by_net, net, {ref}) for net in outputs)
        )
        if complete:
            paths["mcp23017_sixteen_output_expansion"] = True
            counts["gpio_output"] = 16
    complete_gpio_count, complete_gpio_paths = _complete_gpio_paths(complete_gpio_pins, pad_nets)
    complete_gpio_nets = _complete_gpio_nets(complete_gpio_pins, pad_nets)
    if complete_gpio_count:
        counts["complete_mcu_gpio_output"] = complete_gpio_count
    # The parent observer combines these source facts with a physically observed
    # 3.3-V supply and the relevant peripheral/connector proof before publishing
    # any original-brief gate.  GPIO exposure alone cannot prove logger, sensor,
    # battery, or indicator behavior.
    gates: dict[str, str] = {}
    return {
        "net_paths": paths,
        "channel_counts": counts,
        "gates": gates,
        "complete_mcu_refs": sorted(complete_refs),
        "complete_mcu_gpio_paths": complete_gpio_paths,
        "complete_mcu_gpio_nets": complete_gpio_nets,
        "complete_mcu_supplies": complete_mcu_supplies,
    }


_CONNECTOR_PROTOCOL_GATES = frozenset({
    "external_uart_header_binding",
    "external_qwiic_i2c_connector_binding",
    "external_i2c_header_binding",
    "external_spi_header_binding",
})
_CONNECTOR_IO_GATES = frozenset({
    "external_four_screw_terminal_contacts",
    "external_eight_pin_0p1_header",
    "external_two_jst_xh_connectors",
    "external_six_pin_0p1_header",
})
_ANALOG_SENSOR_GATES = frozenset({
    "external_current_sense_sensor_function",
    "external_temperature_sensor_function",
    "external_hall_effect_sensor_function",
    "external_ambient_light_node_function",
})


def _passed_gates(facts: dict[str, Any] | None, allowed: frozenset[str]) -> set[str]:
    gates = (facts or {}).get("gates")
    if not isinstance(gates, dict):
        return set()
    return {gate for gate in allowed if gates.get(gate) == "pass"}


def _powered_complete_mcus(parts, mcu_facts, supply_voltage) -> set[str]:
    """Require the observed supply to remain within the reviewed MCU range."""
    part_by_ref = {str(part.get("ref") or ""): part for part in parts}
    supplies = mcu_facts.get("complete_mcu_supplies")
    if not isinstance(supplies, dict):
        return set()
    powered: set[str] = set()
    for ref in mcu_facts.get("complete_mcu_refs") or ():
        supply = supplies.get(ref)
        part = part_by_ref.get(ref)
        record = _record(part) if isinstance(part, dict) else None
        if not isinstance(supply, dict) or record is None:
            continue
        rail, ground = supply.get("rail"), supply.get("ground")
        if not isinstance(rail, str) or not isinstance(ground, str):
            continue
        voltage = supply_voltage(rail, ground)
        minimum = record.operating_limits.get("supply_min_v", record.operating_limits.get("io_supply_min_v"))
        maximum = record.operating_limits.get("supply_max_v", record.operating_limits.get("io_supply_max_v"))
        if (
            isinstance(voltage, (int, float))
            and not isinstance(voltage, bool)
            and isinstance(minimum, (int, float))
            and isinstance(maximum, (int, float))
            and minimum <= voltage <= maximum
        ):
            powered.add(ref)
    return powered


def _powered_adc_nets(parts, mcu_facts, powered: set[str]) -> set[str]:
    """Return only complete-core nets sourced by reviewed ADC-capable contacts."""
    gpio_nets = mcu_facts.get("complete_mcu_gpio_nets")
    if not isinstance(gpio_nets, dict):
        return set()
    part_by_ref = {str(part.get("ref") or ""): part for part in parts}
    reviewed_adc = reviewed_application_pins("adc")
    adc_nets: set[str] = set()
    for net, sources in gpio_nets.items():
        if not isinstance(net, str) or not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, str) or "." not in source:
                continue
            ref, pin = source.rsplit(".", 1)
            part = part_by_ref.get(ref)
            record = _record(part) if isinstance(part, dict) else None
            port_adc_pins = {
                str(value)
                for name, value in (record.port_pins.items() if record is not None else ())
                if str(name).lower().startswith(("adc", "ain"))
            }
            if (
                ref in powered
                and record is not None
                and pin in reviewed_adc.get(record.identity, set()) | port_adc_pins
            ):
                adc_nets.add(net)
    return adc_nets


def controller_function_gates(
    parts,
    names,
    pad_nets,
    *,
    state,
    mcu_facts,
    connector_facts,
    analog_facts,
    outputs,
    supply_voltage,
) -> dict[str, str]:
    """Compose controller gates only from complete powered circuit evidence.

    The caller supplies ``supply_voltage`` from its board-bound power observer;
    this module never accepts a rail label as voltage evidence.  There is no
    positive logger, battery, or indicator claim here: those functions need
    storage/source/display proofs not represented by an MCU or connector alone.
    """
    del names, pad_nets, state, outputs
    if not isinstance(mcu_facts, dict):
        return {}
    powered = _powered_complete_mcus(parts, mcu_facts, supply_voltage)
    if not powered:
        return {}
    protocol = _passed_gates(connector_facts, _CONNECTOR_PROTOCOL_GATES)
    io_connectors = _passed_gates(connector_facts, _CONNECTOR_IO_GATES)
    powered_adc_nets = _powered_adc_nets(parts, mcu_facts, powered)
    external_gpio_paths = mcu_facts.get("complete_mcu_gpio_paths")
    powered_external_gpio_nets = {
        net for net, sources in (
            external_gpio_paths.items() if isinstance(external_gpio_paths, dict) else ()
        )
        if isinstance(sources, list)
        and any(source.split(".", 1)[0] in powered for source in sources if isinstance(source, str))
    }
    gates: dict[str, str] = {}
    # Programming/recovery plus independently exposed application I/O makes a
    # development board; a separately requested protocol has its own obligation.
    if len(powered_external_gpio_nets) >= 2:
        gates["external_development_board_function"] = "pass"
    # An interface board needs a proven protocol and bound I/O connector form.
    if protocol and io_connectors:
        gates["external_interface_board_function"] = "pass"
    sensor_gates = _passed_gates(analog_facts, _ANALOG_SENSOR_GATES)
    analog_nets = (analog_facts or {}).get("analog_output_nets")
    if (
        sensor_gates
        and protocol
        and isinstance(analog_nets, list)
        and any(isinstance(net, str) and net in powered_adc_nets for net in analog_nets)
    ):
        gates["external_sensor_breakout_function"] = "pass"
    return gates
