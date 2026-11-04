"""Artifact-bound evidence for the host-plus-battery 5 V power path.

This module intentionally returns no result unless every asserted item is bound
to a delivered physical inventory record and the physical pad graph proves the
manufacturer-shaped circuit.  It does not inspect architecture rail names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.validation import (
    _R_INFIX_RE, _R_VALUE_RE, _capacitance_farads, _resistance_ohms,
)

SourceClass = Literal["host_5v", "single_cell_liion"]
SourceVoltageVerifier = Callable[[dict[str, Any], str, str, SourceClass, float, float], bool]


@dataclass(frozen=True)
class DualSourcePowerEvidence:
    """A physical mux output the caller may subsequently bind to a load."""

    output_net: str
    ground_net: str
    output_voltage_v: float
    minimum_output_voltage_v: float
    maximum_output_voltage_v: float
    continuous_load_a: float


def _record(part: dict[str, Any]):
    return physical_inventory_record(
        mpn=part.get("mpn"), symbol=part.get("symbol"),
        footprint=part.get("footprint"), datasheet=part.get("datasheet"),
        sourcing_note=part.get("sourcing_note"),
    )


def _pads(ref: str, pad_nets: dict[tuple[str, str], str]) -> dict[str, str]:
    return {pin: net for (part, pin), net in pad_nets.items() if part == ref and net}


def _exact(parts: list[dict[str, Any]], identity: str) -> list[tuple[dict[str, Any], Any]]:
    return [(part, record) for part in parts if (record := _record(part)) is not None and record.identity == identity]


def _two_terminal(
    parts: list[dict[str, Any]], pad_nets: dict[tuple[str, str], str], *,
    kind: Literal["R", "C", "L", "D"], left: str, right: str, value: float | None = None,
    rel_tol: float = 0.01, mpn: str | None = None,
) -> list[dict[str, Any]]:
    """Only delivered two-terminal components of the requested kind/value count."""
    from .electrical_artifact_evidence import _symbol_kind

    if not left or not right or left == right:
        return []
    found: list[dict[str, Any]] = []
    for part in parts:
        ref = str(part.get("ref") or "")
        if _symbol_kind(part) != {"R": "resistor", "C": "capacitor", "L": "inductor", "D": "diode"}[kind]:
            continue
        pins = _pads(ref, pad_nets)
        if len(pins) != 2 or set(pins.values()) != {left, right}:
            continue
        record = _record(part)
        if mpn is not None and (
            str(part.get("mpn") or "").casefold() != mpn.casefold()
            or record is None or record.identity != mpn.casefold()
        ):
            continue
        if record is not None and "polarized-capacitor" in record.physical_features:
            if (
                pins.get(str(record.port_pins.get("positive"))) != left
                or pins.get(str(record.port_pins.get("negative"))) != right
            ):
                continue
        if kind == "D" and (
            record is None
            or pins.get(str(record.port_pins.get("anode"))) != left
            or pins.get(str(record.port_pins.get("cathode"))) != right
        ):
            continue
        if value is not None:
            measured = (
                _resistance_ohms(str(part.get("value") or "")) if kind == "R"
                else _capacitance_farads(str(part.get("value") or "")) if kind == "C"
                else None
            )
            if measured is None or abs(measured - value) > value * rel_tol:
                continue
        found.append(part)
    return found


def _series_peer(
    parts: list[dict[str, Any]], pad_nets: dict[tuple[str, str], str], *,
    kind: Literal["R", "C"], net: str, value: float, rel_tol: float = 0.01,
) -> set[str]:
    from .electrical_artifact_evidence import _symbol_kind

    peers: set[str] = set()
    for part in parts:
        ref = str(part.get("ref") or "")
        if _symbol_kind(part) != {"R": "resistor", "C": "capacitor"}[kind]:
            continue
        pins = _pads(ref, pad_nets)
        if len(pins) != 2 or net not in pins.values():
            continue
        measured = (
            _resistance_ohms(str(part.get("value") or "")) if kind == "R"
            else _capacitance_farads(str(part.get("value") or ""))
        )
        if measured is None or abs(measured - value) > value * rel_tol:
            continue
        peers.update(value for value in pins.values() if value != net)
    return peers


def _record_port(ref: str, record: Any, port: str, pads: dict[tuple[str, str], str]) -> str | None:
    pin = str(record.port_pins.get(port) or "")
    return pads.get((ref, pin)) if pin else None


def observe_dual_source_battery_power(
    parts: list[dict[str, Any]],
    pad_nets: dict[tuple[str, str], str],
    *,
    source_voltage: SourceVoltageVerifier,
) -> DualSourcePowerEvidence | None:
    """Prove the physical dual-source stage and return its independent output.

    The callback is responsible only for the externally specified source
    interface ranges; it receives the actual connector component and actual
    positive/return nets. It cannot bless an arbitrary rail name.
    """
    muxes = _exact(parts, "tps2113apwr")
    boosts = _exact(parts, "tps61088rhlr")
    batteries = _exact(parts, "s2b-ph-sm4-tb(lf)(sn)")
    jumpers = _exact(parts, "tsw-102-07-g-s")
    headers = [
        (part, record) for part in parts
        if (record := _record(part)) is not None
        and ("pin-header" in record.physical_features or "header" in record.physical_features)
        and {"1", "2"} <= set(record.contacts)
    ]
    if len(muxes) != 1 or len(boosts) != 1 or len(batteries) != 1:
        return None
    mux, mux_record = muxes[0]
    boost, boost_record = boosts[0]
    mux_ref, boost_ref = str(mux["ref"]), str(boost["ref"])
    mux_pads, boost_pads = _pads(mux_ref, pad_nets), _pads(boost_ref, pad_nets)

    host = _record_port(mux_ref, mux_record, "input1", pad_nets)
    battery_5v = _record_port(mux_ref, mux_record, "input2", pad_nets)
    output = _record_port(mux_ref, mux_record, "output", pad_nets)
    ground = _record_port(mux_ref, mux_record, "ground", pad_nets)
    enable = _record_port(mux_ref, mux_record, "enable", pad_nets)
    vsns = _record_port(mux_ref, mux_record, "source_select", pad_nets)
    limit = _record_port(mux_ref, mux_record, "limit", pad_nets)
    battery_raw = _record_port(boost_ref, boost_record, "input", pad_nets)
    switch = _record_port(boost_ref, boost_record, "switch", pad_nets)
    feedback = _record_port(boost_ref, boost_record, "feedback", pad_nets)
    bootstrap = _record_port(boost_ref, boost_record, "bootstrap", pad_nets)
    vcc = _record_port(boost_ref, boost_record, "vcc", pad_nets)
    compensation = _record_port(boost_ref, boost_record, "compensation", pad_nets)
    boost_ground = _record_port(boost_ref, boost_record, "analog_ground", pad_nets)
    frequency, soft_start, boost_limit = (boost_pads.get(pin) for pin in ("3", "10", "19"))
    separate_nets = (
        host, battery_raw, battery_5v, output, ground, switch, enable, vsns, limit,
        feedback, bootstrap, vcc, compensation, frequency, soft_start, boost_limit,
    )
    if (
        not all(separate_nets)
        or ground != boost_ground
        or len(set(separate_nets)) != len(separate_nets)
        or any(boost_pads.get(str(pin)) != switch for pin in range(4, 8))
        or any(boost_pads.get(str(pin)) != battery_5v for pin in range(14, 17))
        or boost_pads.get("20") != ground
        or boost_pads.get("21") != ground
        or boost_pads.get("2") != battery_raw
        or boost_pads.get("13") != ground
        or boost_pads.get("11") or boost_pads.get("12")
    ):
        return None

    # Net names alone do not prove isolation: delivered low-resistance links
    # can bypass the mux, its boost inductor, or short a supply to ground.
    from .electrical_artifact_evidence import _two_terminal as delivered_terminals

    links: dict[str, set[str]] = {}
    for resistor, left, right in delivered_terminals(parts, pad_nets, "R"):
        value = str(resistor.get("value") or "").strip()
        resistance = _resistance_ohms(value)
        # Zero is intentionally rejected by the shared current-limiter parser.
        zero = _R_INFIX_RE.match(value)
        if zero is not None:
            is_zero = int(zero.group(1)) == 0 and int(zero.group(3)) == 0
        else:
            zero = _R_VALUE_RE.match(value)
            is_zero = zero is not None and float(zero.group(1)) == 0
        if is_zero or (resistance is not None and resistance <= 0.1):
            links.setdefault(left[1], set()).add(right[1])
            links.setdefault(right[1], set()).add(left[1])
    power_nets = {host, battery_raw, battery_5v, output, ground, switch}
    for net in power_nets:
        visited = {net}
        pending = [net]
        while pending:
            for peer in links.get(pending.pop(), ()):
                if peer in power_nets and peer != net:
                    return None
                if peer not in visited:
                    visited.add(peer)
                    pending.append(peer)

    battery_part, battery_record = batteries[0]
    battery_pads = _pads(str(battery_part["ref"]), pad_nets)
    battery_positive = battery_pads.get("1")
    battery_return = battery_pads.get("2")
    jumpers = [
        (part, record) for part, record in jumpers
        if _pads(str(part["ref"]), pad_nets).get("1") == enable
    ]
    if len(jumpers) != 1:
        return None
    jumper_part, _jumper_record = jumpers[0]
    enable_high = _pads(str(jumper_part["ref"]), pad_nets).get("2")
    if not enable_high or enable_high in {enable, host, battery_raw, battery_5v, output, ground, switch}:
        return None
    host_candidates = [
        (part, record) for part, record in headers
        if str(part["ref"]) != str(jumper_part["ref"])
        and _pads(str(part["ref"]), pad_nets).get("1") == host
        and _pads(str(part["ref"]), pad_nets).get("2") == ground
    ]
    if (
        battery_positive != battery_raw or battery_return != ground or len(host_candidates) != 1
        or not source_voltage(host_candidates[0][0], host, ground, "host_5v", 4.75, 5.25)
        or not source_voltage(battery_part, battery_raw, ground, "single_cell_liion", 3.0, 4.2)
    ):
        return None

    # The exact polymer bank avoids an output MLCC DC-bias inference.
    # Initial +20 C tolerance gives 240 uF minimum with the initial ESR
    # bound; do not combine aged capacitance with an unaged ESR guarantee.
    support = boost_record.support_network.get("artifact_requirements", {})
    profile = support.get("dual_source_5v", {}) if isinstance(support, dict) else {}
    effective_uf = profile.get("guaranteed_capacitance_min_uf") if isinstance(profile, dict) else None
    required_uf = profile.get("required_capacitance_min_uf") if isinstance(profile, dict) else None
    max_esr = profile.get("maximum_esr_ohm") if isinstance(profile, dict) else None
    if (
        profile.get("effective_capacitance_status") != "bounded_20c_initial_delivery_mpn"
        or profile.get("mpn") != "6TPE100MAZB"
        or not isinstance(effective_uf, (int, float))
        or not isinstance(required_uf, (int, float))
        or not isinstance(max_esr, (int, float))
        or required_uf < 6.8
        or effective_uf < required_uf
        or max_esr > 0.012
    ):
        return None
    comp_nodes = _series_peer(
        parts, pad_nets, kind="R", net=compensation, value=69800.0, rel_tol=0.02,
    )
    if not (
        len(_two_terminal(parts, pad_nets, kind="L", left=battery_raw, right=switch, mpn="xal7070-222mec")) == 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=battery_raw, right=ground, value=22e-6, mpn="emk316bb7226ml-t")) >= 2
        and len(_two_terminal(parts, pad_nets, kind="C", left=battery_raw, right=ground, value=100e-9, mpn="cl10b104ko8nnnc")) >= 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=vcc, right=ground, value=2.2e-6, mpn="cl10a225ko8nnnc")) == 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=bootstrap, right=switch, value=100e-9, mpn="cl10b104ko8nnnc")) == 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=soft_start, right=ground, value=47e-9, mpn="cl10b473kb8nnnc")) == 1
        and len(_two_terminal(parts, pad_nets, kind="R", left=frequency, right=switch, value=301000.0)) == 1
        and len(_two_terminal(parts, pad_nets, kind="R", left=boost_limit, right=ground, value=49900.0)) == 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=battery_5v, right=ground, value=100e-6, mpn="6tpe100mazb")) == 3
        and len(_two_terminal(parts, pad_nets, kind="R", left=battery_5v, right=feedback, value=178000.0)) == 1
        and len(_two_terminal(parts, pad_nets, kind="R", left=feedback, right=ground, value=56000.0)) == 1
        and len(comp_nodes) == 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=next(iter(comp_nodes)), right=ground, value=15e-9, mpn="c2012c0g1h153jt000n")) == 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=compensation, right=ground, value=33e-12, mpn="cl10c330jb8nnnc")) == 1
    ):
        return None

    # The TSW header is a physical disable shunt; EN is otherwise pulled low.
    # The host header itself may be any reviewed, physically bound 2.54-mm form.
    features = mux_record.physical_features
    if not (
        {"power-mux", "reverse-conduction-blocking", "cross-conduction-blocking", "power-source-isolation"} <= features
        and len(_two_terminal(parts, pad_nets, kind="D", left=host, right=enable_high, mpn="ss14")) == 1
        and len(_two_terminal(parts, pad_nets, kind="D", left=battery_5v, right=enable_high, mpn="ss14")) == 1
        and len(_two_terminal(parts, pad_nets, kind="R", left=enable, right=ground, value=100000.0)) == 1
        and len(_two_terminal(parts, pad_nets, kind="R", left=host, right=vsns, value=100000.0)) == 1
        and len(_two_terminal(parts, pad_nets, kind="R", left=vsns, right=ground, value=27000.0)) == 1
        and len(_two_terminal(parts, pad_nets, kind="R", left=limit, right=ground, value=400.0)) == 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=host, right=ground, value=100e-9, mpn="cl10b104ko8nnnc")) >= 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=battery_5v, right=ground, value=100e-9, mpn="cl10b104ko8nnnc")) >= 1
        and len(_two_terminal(parts, pad_nets, kind="C", left=output, right=ground, value=22e-6, mpn="emk316bb7226ml-t")) >= 1
    ):
        return None
    return DualSourcePowerEvidence(output, ground, 5.0, 4.675, 5.25, 0.5)
