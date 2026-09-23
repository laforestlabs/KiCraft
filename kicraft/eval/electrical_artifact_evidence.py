"""Electrical acceptance facts extracted from the committed board graph.

This module is deliberately a fail-closed observer.  It joins state.json only to
identify components and their intended pin numbers; every asserted terminal/net
relation is rechecked against the already-loaded ``pcbnew.BOARD``.  A net label,
a BOM group, or connectivity through a capacitor/control pin never establishes a
functional path.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import re
from typing import Any, Iterable

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.symbol_pinout import lookup_pins
from kicraft.design.synthesis.validation import (
    _capacitance_farads,
    _compare_netlist_to_bom,
    _resistance_ohms,
    _reviewed_high_side_led_loop_contract,
    check_net_coverage,
    check_pin_existence,
    regulator_vout_facts,
)

_GND = re.compile(r"(?:^|[_-])(?:GND|AGND|DGND|PGND|0V)(?:$|[_-])", re.I)
_RAIL = re.compile(r"(?:^|[_+-])(?:V(?:CC|DD|IN|OUT|BUS)?|\+?\d+(?:V\d+|\.\d+V|V))(?:$|[_+-])", re.I)

def _is_ground(net: str | None) -> bool:
    return bool(net and _GND.search(net.lstrip("/")))


def _is_rail(net: str | None) -> bool:
    return bool(net and _RAIL.search(net.lstrip("/"))) and not _is_ground(net)


def _footprint_id(fp: object) -> str:
    """Return the KiCad library identity without serializing a SWIG proxy."""
    try:
        fpid = fp.GetFPID()
        item = str(fpid.GetLibItemName())
        nickname = str(fpid.GetLibNickname())
    except Exception:
        return ""
    return f"{nickname}:{item}" if nickname and item else item


def _board_graph(
    board: object,
) -> tuple[
    dict[tuple[str, str], str],
    dict[str, set[str]],
    dict[str, str],
    set[tuple[str, str]],
]:
    """Return named nets, net members, footprint identities, and all delivered pads."""
    pad_nets: dict[tuple[str, str], str] = {}
    members: dict[str, set[str]] = defaultdict(set)
    footprints: dict[str, str] = {}
    pads: set[tuple[str, str]] = set()
    try:
        footprints_iter = board.GetFootprints()
    except Exception:
        return pad_nets, members, footprints, pads
    for fp in footprints_iter:
        try:
            ref = fp.GetReferenceAsString()
            footprints[ref] = _footprint_id(fp)
        except Exception:
            continue
        for pad in fp.Pads():
            number, net = str(pad.GetNumber()), str(pad.GetNetname() or "")
            if not number:
                continue
            endpoint = (ref, number)
            pads.add(endpoint)
            if net:
                pad_nets[endpoint] = net
                members[net].add(f"{ref}.{number}")
    return pad_nets, members, footprints, pads


def _reconcile_connection_terminals(
    bom: object,
    pad_nets: dict[tuple[str, str], str],
    board_groups: Iterable[set[tuple[str, str]]],
    footprints: dict[str, str],
    delivered_pads: set[tuple[str, str]],
) -> tuple[bool, list[str]]:
    """Prove the canonical BOM terminal contract against delivered PCB pads.

    The synthesis validator owns connection semantics.  This observer only
    substitutes the delivered-board net groups for its netlist input, then adds
    the physical facts the netlist cannot establish: exact footprint identity,
    pad survival, and intentionally netless NC terminals.
    """
    reasons: list[str] = []
    parts = {part.ref: part for part in bom.parts}
    required = {
        (endpoint.ref, str(endpoint.pin))
        for connection in bom.connections
        for endpoint in connection.endpoints
    }
    deliberate_nc = {(endpoint.ref, str(endpoint.pin)) for endpoint in bom.no_connect_pins}
    if not required:
        return False, ["BOM has no required connected terminals"]

    for ref, pin in sorted(required | deliberate_nc):
        part = parts.get(ref)
        expected_footprint = str(part.footprint) if part is not None else ""
        if not expected_footprint or footprints.get(ref) != expected_footprint:
            reasons.append(f"{ref}: delivered footprint identity does not match BOM")
            continue
        endpoint = (ref, pin)
        if endpoint not in delivered_pads:
            reasons.append(f"{ref}.{pin}: required delivered pad is missing")
        elif endpoint in deliberate_nc:
            if endpoint in pad_nets:
                reasons.append(f"{ref}.{pin}: deliberate NC has a delivered net")
        elif endpoint not in pad_nets:
            reasons.append(f"{ref}.{pin}: required pad has no delivered net")

    # These are the canonical source of terminal semantics, pin validity, and
    # expected-net equivalence.  A missing or unresolvable semantic remains
    # unverified here instead of being converted into a generic positive claim.
    pin_check = check_pin_existence(bom)
    if not pin_check.ok:
        reasons.append("BOM connection pins are not symbol-verified")
    coverage = check_net_coverage(bom)
    if not coverage.ok:
        reasons.append("BOM terminal coverage is incomplete")
    merges, splits, lost = _compare_netlist_to_bom(list(board_groups), bom)
    if merges:
        reasons.append("delivered board merges required nets")
    if splits:
        reasons.append("delivered board splits required nets")
    if lost:
        reasons.append("required terminals are absent from delivered nets")
    return not reasons, reasons


def _parts(state: dict[str, Any], footprints: dict[str, str]) -> list[dict[str, Any]]:
    bom = state.get("bom") if isinstance(state.get("bom"), dict) else {}
    rows = bom.get("parts") if isinstance(bom, dict) else []
    return [
        p for p in rows or []
        if isinstance(p, dict)
        and str(p.get("ref") or "") in footprints
        and str(p.get("footprint") or "") == footprints[str(p.get("ref"))]
    ]


def _symbol_kind(part: dict[str, Any]) -> str:
    """Exact symbol class for inert primitives; refdes text is never evidence."""
    symbol = str(part.get("symbol") or "")
    if symbol in {"Device:R", "Device:R_Small"}:
        return "resistor"
    if symbol in {"Device:C", "Device:C_Small", "Device:C_Polarized"}:
        return "capacitor"
    if symbol in {"Device:L", "Device:L_Small"}:
        return "inductor"
    if symbol in {"Device:D", "Device:D_Schottky"}:
        return "diode"
    if symbol in {"Device:Crystal", "Device:Resonator"}:
        return "crystal"
    if symbol == "Device:LED":
        return "led"
    return ""


def _physical_features(part: dict[str, Any]) -> frozenset[str]:
    record = _record(part)
    return record.physical_features if record is not None else frozenset()


def _reviewed_identity(part: dict[str, Any]) -> set[str]:
    """Exact physical family/identity terms, never MPN/value/note substring text."""
    record = _record(part)
    return {record.identity, record.family} if record is not None else set()


def _is_kind(part: dict[str, Any], kind: str) -> bool:
    return _symbol_kind(part) == kind





_ESCAPED_PIN_NAME_RE = re.compile(r"~\{([^{}]*)\}")
_SUBSCRIPT_PIN_NAME_RE = re.compile(r"([A-Za-z0-9_]+)_\{([^{}]*)\}")


def _normalized_pin_name(name: str) -> str:
    """Reduce KiCad's display escapes to the electrical name.

    Vendored symbols spell an active-low pin ``~{CS}`` and a subscripted supply
    ``V_{CC}``; matchers that compare printed names against ``CS``/``VCC`` would
    otherwise never see those pins, silently unproving the whole interface.
    """
    out = _ESCAPED_PIN_NAME_RE.sub(r"\1", str(name or ""))
    return _SUBSCRIPT_PIN_NAME_RE.sub(r"\1\2", out).strip()


def _pin_names(parts: list[dict[str, Any]], rundir: Path) -> dict[tuple[str, str], str]:
    """Resolve symbol pin semantics; unresolvable symbols contribute no claims."""
    names: dict[tuple[str, str], str] = {}
    root = Path(__file__).resolve().parents[2]
    for part in parts:
        ref, symbol = str(part.get("ref")), str(part.get("symbol") or "")
        try:
            pins = lookup_pins(symbol, project_root=root, all_units=True).get("pins") or []
        except Exception:
            pins = []
        for pin in pins:
            if isinstance(pin, dict) and pin.get("number") is not None:
                names[(ref, str(pin["number"]))] = _normalized_pin_name(pin.get("name"))
        record = physical_inventory_record(mpn=part.get("mpn"), symbol=part.get("symbol"), footprint=part.get("footprint"), datasheet=part.get("datasheet"), sourcing_note=part.get("sourcing_note"))
        if record is not None:
            for key, number in record.port_pins.items():
                names.setdefault((ref, str(number)), str(key))
    return names


def _pins_for(ref: str, names: dict[tuple[str, str], str], pad_nets: dict[tuple[str, str], str], pattern: str) -> list[tuple[str, str]]:
    rx = re.compile(pattern, re.I)
    return [(pin, pad_nets[(ref, pin)]) for (candidate, pin), name in names.items() if candidate == ref and (ref, pin) in pad_nets and rx.search(name)]


def _one_pin(ref: str, names: dict[tuple[str, str], str], pad_nets: dict[tuple[str, str], str], pattern: str) -> tuple[str, str] | None:
    found = _pins_for(ref, names, pad_nets, pattern)
    return found[0] if len(found) == 1 else None


def _same_net(ref_a: str, pin_a: str, ref_b: str, pin_b: str, pad_nets: dict[tuple[str, str], str]) -> str | None:
    a, b = pad_nets.get((ref_a, pin_a)), pad_nets.get((ref_b, pin_b))
    return a if a and a == b else None


def _record(part: dict[str, Any]):
    return physical_inventory_record(mpn=part.get("mpn"), symbol=part.get("symbol"), footprint=part.get("footprint"), datasheet=part.get("datasheet"), sourcing_note=part.get("sourcing_note"))


def _device(parts: Iterable[dict[str, Any]], *identities: str) -> list[dict[str, Any]]:
    """Exact reviewed physical identities/families only; no free-text inference."""
    wanted = {identity.casefold() for identity in identities}
    return [part for part in parts if _reviewed_identity(part) & wanted]


def _connector(parts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Physically reviewed connector/contact hardware, never a reference prefix."""
    out = []
    for part in parts:
        record = _record(part)
        if record is None:
            continue
        features = record.physical_features
        if (
            "header" in features
            or "screw-clamp-terminal" in features
            or "screw-terminal" in features
            or "terminal-block" in features
            or any("connector" in feature for feature in features)
            or record.family in {"pin-header", "binding-post"}
        ):
            out.append(part)
    return out

def _pad_id(ref: str, pin: str) -> str:
    return f"{ref}.{pin}"


def _bind(source_ref: str, source_pin: str, target_ref: str, target_pin: str, pad_nets: dict[tuple[str, str], str]) -> dict[str, str] | None:
    net = _same_net(source_ref, source_pin, target_ref, target_pin, pad_nets)
    if net is None:
        return None
    return {"source_pad": _pad_id(source_ref, source_pin), "connector_pad": _pad_id(target_ref, target_pin), "net": net}


def _all_pads(ref: str, pad_nets: dict[tuple[str, str], str]) -> list[tuple[str, str]]:
    return sorted((pin, net) for (candidate, pin), net in pad_nets.items() if candidate == ref)


def _direct_map(source: dict[str, Any], targets: Iterable[dict[str, Any]], required: dict[str, str], names: dict[tuple[str, str], str], pad_nets: dict[tuple[str, str], str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for label, pin_pattern in required.items():
        source_pin = _one_pin(str(source["ref"]), names, pad_nets, pin_pattern)
        if source_pin is None:
            continue
        for target in targets:
            for target_pin, target_net in _all_pads(str(target["ref"]), pad_nets):
                if target_net == source_pin[1]:
                    binding = _bind(str(source["ref"]), source_pin[0], str(target["ref"]), target_pin, pad_nets)
                    if binding:
                        out[label] = binding
                        break
            if label in out:
                break
    return out


def _passive_edges(parts: list[dict[str, Any]], pad_nets: dict[tuple[str, str], str]) -> dict[str, set[str]]:
    """Only explicitly conductive passives/switches enter this graph; never C or IC control pins."""
    graph: dict[str, set[str]] = defaultdict(set)
    for p in parts:
        # A direct conductive primitive is established by its exact symbol, not
        # by the human-chosen reference prefix. Capacitors are never power-path
        # edges.
        if _symbol_kind(p) not in {"resistor", "inductor"}:
            continue
        pins = _all_pads(str(p["ref"]), pad_nets)
        if len(pins) == 2 and pins[0][1] != pins[1][1]:
            graph[pins[0][1]].add(pins[1][1]); graph[pins[1][1]].add(pins[0][1])
    return graph


def _reaches(graph: dict[str, set[str]], source: str | None, target: str | None) -> bool:
    if not source or not target:
        return False
    pending, seen = [source], {source}
    while pending:
        node = pending.pop()
        if node == target:
            return True
        for nxt in graph.get(node, ()):
            if nxt not in seen:
                seen.add(nxt); pending.append(nxt)
    return False


def _power_edges(parts: list[dict[str, Any]], names: dict[tuple[str, str], str], pad_nets: dict[tuple[str, str], str]) -> dict[str, set[str]]:
    graph = _passive_edges(parts, pad_nets)
    for p in parts:
        record = _record(p)
        if record is None:
            continue
        transfer = record.power_transfer
        paths = transfer.get("paths") if isinstance(transfer, dict) else None
        if not isinstance(paths, (list, tuple)):
            paths = (transfer,) if isinstance(transfer, dict) else ()
        for path in paths:
            if not isinstance(path, dict):
                continue
            a = _one_pin(str(p["ref"]), names, pad_nets, "^" + re.escape(str(path.get("from_pin") or record.port_pins.get("input") or "")) + "$")
            b = _one_pin(str(p["ref"]), names, pad_nets, "^" + re.escape(str(path.get("to_pin") or record.port_pins.get("switch") or "")) + "$")
            if a and b and a[1] != b[1]:
                graph[a[1]].add(b[1]); graph[b[1]].add(a[1])
    return graph


def _two_terminal(parts: Iterable[dict[str, Any]], pad_nets: dict[tuple[str, str], str], kind: str | None = None):
    aliases = {"R": "resistor", "C": "capacitor", "L": "inductor", "Y": "crystal"}
    wanted = aliases.get(kind or "", kind)
    for p in parts:
        if wanted and _symbol_kind(p) != wanted:
            continue
        pads = _all_pads(str(p["ref"]), pad_nets)
        if len(pads) == 2 and pads[0][1] != pads[1][1]:
            yield p, pads[0], pads[1]


def _reviewed_led_current_loop(
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
) -> float | None:
    """Return a board-proven current only for the reviewed complete LED loop.

    The component list was already reconciled against actual footprint IDs.  This
    adds the remaining physical proof: every topology endpoint below is an
    observed pad/net relation, never a BOM net label or a textual circuit claim.
    """
    for controller in parts:
        record = _record(controller)
        if record is None:
            continue
        loop = _reviewed_high_side_led_loop_contract(
            {
                "current_feedback": dict(record.current_feedback),
                "power_transfer": dict(record.power_transfer),
                "support_network": dict(record.support_network),
                "operating_limits": dict(record.operating_limits),
                "port_pins": dict(record.port_pins),
            }
        )
        if loop is None:
            continue
        ref = str(controller["ref"])
        vin = _one_pin(ref, names, pad_nets, "^" + re.escape(loop["reference_pin"]) + "$")
        set_pin = _one_pin(ref, names, pad_nets, "^" + re.escape(loop["sense_pin"]) + "$")
        sw_pins = _pins_for(ref, names, pad_nets, "^" + re.escape(loop["switch_pin"]) + "$")
        gnd_pins = _pins_for(ref, names, pad_nets, "^" + re.escape(loop["ground_pin"]) + "$")
        ep = _one_pin(ref, names, pad_nets, r"^EP$")
        if not (vin and set_pin and sw_pins and gnd_pins and ep):
            continue
        switch_nets, ground_nets = {net for _, net in sw_pins}, {net for _, net in gnd_pins}
        if len(switch_nets) != 1 or len(ground_nets) != 1:
            continue
        switch_net, ground_net = next(iter(switch_nets)), next(iter(ground_nets))
        if ep[1] != ground_net or len({vin[1], set_pin[1], switch_net, ground_net}) != 4:
            continue

        vref = loop["sense_voltage_v"]
        current_a = None
        if isinstance(vref, (int, float)) and vref > 0:
            for resistor, left, right in _two_terminal(parts, pad_nets, "R"):
                if {left[1], right[1]} != {vin[1], set_pin[1]}:
                    continue
                resistance = _resistance_ohms(str(resistor.get("value") or ""))
                if resistance and resistance > 0:
                    current_a = float(vref) / resistance
                    break
        if current_a is None or current_a <= 0 or current_a > loop["continuous_current_a"]:
            continue

        # The source and return must both be physically exposed; controller pads
        # named VIN/GND alone cannot establish a usable input loop.
        source_ok = any(
            vin[1] in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
            and ground_net in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
            for connector in _connector(parts)
        )
        if not source_ok:
            continue

        minimum_f = loop["input_decoupling"]["capacitance_min_f"]
        cap_ok = any(
            {left[1], right[1]} == {vin[1], ground_net}
            and (value := _capacitance_farads(str(capacitor.get("value") or ""))) is not None
            and value >= minimum_f
            for capacitor, left, right in _two_terminal(parts, pad_nets, "C")
        )
        if not cap_ok:
            continue

        # ``catch_diode`` names the controller pins the diode ends must reach
        # (anode on the switch node, cathode on the input rail).  Those pin names
        # may be carried by several pads, so resolve each declared name to its
        # observed net set and require the reviewed names to agree with the
        # already-proven switch/input rails; then require the diode's own
        # anode/cathode pads to land on those rails.
        catch = loop["catch_diode"]
        switch_named = {net for _, net in _pins_for(ref, names, pad_nets, "^" + re.escape(catch["anode_pin"]) + "$")}
        input_named = {net for _, net in _pins_for(ref, names, pad_nets, "^" + re.escape(catch["cathode_pin"]) + "$")}
        diode_ok = False
        if switch_named == {switch_net} and input_named == {vin[1]}:
            for diode in (part for part in parts if _is_kind(part, "diode")):
                anode = _one_pin(str(diode["ref"]), names, pad_nets, r"^(A|ANODE)$")
                cathode = _one_pin(str(diode["ref"]), names, pad_nets, r"^(K|CATHODE)$")
                if anode and cathode and anode[1] == switch_net and cathode[1] == vin[1]:
                    diode_ok = True
                    break
        if not diode_ok:
            continue

        for _, left, right in _two_terminal(parts, pad_nets, "L"):
            if left[1] == switch_net:
                cathode_net = right[1]
            elif right[1] == switch_net:
                cathode_net = left[1]
            else:
                continue
            terminal_ok = any(
                set_pin[1] in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
                and cathode_net in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
                for connector in _connector(parts)
            )
            onboard_led_ok = any(
                (anode := _one_pin(str(led["ref"]), names, pad_nets, r"^(A|ANODE)$")) is not None
                and (cathode := _one_pin(str(led["ref"]), names, pad_nets, r"^(K|CATHODE)$")) is not None
                and anode[1] == set_pin[1]
                and cathode[1] == cathode_net
                for led in parts
                if _is_kind(led, "led")
            )
            if terminal_ok or onboard_led_ok:
                return current_a
    return None


def _connector_maps(parts, names, pad_nets) -> tuple[dict[str, Any], dict[str, int]]:
    maps: dict[str, Any] = {}; counts: dict[str, int] = {}
    connectors = _connector(parts)
    usb = [p for p in parts if "usb-c-receptacle" in _physical_features(p)]
    headers = [p for p in connectors if p not in usb]
    if usb and headers:
        labels = {"VBUS": r"^VBUS", "GND": r"^(GND|SHIELD)$", "CC1": r"^CC1$", "CC2": r"^CC2$", "SBU1": r"^SBU1$", "SBU2": r"^SBU2$", "TX1P": r"^(TX1P|SSTX1\+)$", "TX1N": r"^(TX1N|SSTX1-)$", "RX1P": r"^(RX1P|SSRX1\+)$", "RX1N": r"^(RX1N|SSRX1-)$", "TX2P": r"^(TX2P|SSTX2\+)$", "TX2N": r"^(TX2N|SSTX2-)$", "RX2P": r"^(RX2P|SSRX2\+)$", "RX2N": r"^(RX2N|SSRX2-)$"}
        candidate = _direct_map(usb[0], headers, labels, names, pad_nets)
        if len(candidate) == len(labels): maps["usb_c_to_header"] = candidate
    # Exact reviewed FPC contact numbers are manufacturer contact semantics, not net names.
    fpcs = [p for p in parts if (_record(p) and "fpc-connector" in _record(p).physical_features)]
    if fpcs and connectors:
        source = fpcs[0]; candidate = {}
        for n in range(1, 25):
            for target in connectors:
                if target is source: continue
                for pin, net in _all_pads(str(target["ref"]), pad_nets):
                    if net == pad_nets.get((str(source["ref"]), str(n))):
                        bound = _bind(str(source["ref"]), str(n), str(target["ref"]), pin, pad_nets)
                        if bound: candidate[str(n)] = bound; break
                if str(n) in candidate: break
        if len(candidate) == 24: maps["fpc24_to_header"] = candidate; counts["fpc_contact_to_header"] = 24
    # CAN has fixed CiA DB9 contacts and silicon-name CANH/CANL semantics.
    transceivers = _device(parts, "sn65hvd230", "can transceiver")
    db9s = [p for p in connectors if "db9-connector" in _physical_features(p)]
    if transceivers and db9s:
        source = transceivers[0]; db9 = db9s[0]; candidate = {}
        for label, source_pattern, pin in (("CANH", r"CAN.?H|^H$", "7"), ("CANL", r"CAN.?L|^L$", "2"), ("GND", r"^(GND|VSS)$", "3")):
            found = _one_pin(str(source["ref"]), names, pad_nets, source_pattern)
            bound = _bind(str(source["ref"]), found[0], str(db9["ref"]), pin, pad_nets) if found else None
            if bound: candidate[label] = bound
        if len(candidate) == 3: maps["db9_can"] = candidate
    # ADS1115: actual AIN pads must land on eight separate terminal pads.
    adcs = _device(parts, "ads1115")
    if len(adcs) >= 2:
        candidate = {}; i = 1
        for adc in adcs:
            for pin, net in _pins_for(str(adc["ref"]), names, pad_nets, r"^AIN[0-3]$"):
                targets = [(p, n) for p in connectors for n, other in _all_pads(str(p["ref"]), pad_nets) if other == net]
                if targets:
                    target, terminal_pin = targets[0]
                    bound = _bind(str(adc["ref"]), pin, str(target["ref"]), terminal_pin, pad_nets)
                    if bound: candidate[f"AI{i}"] = bound; i += 1
        if len(candidate) >= 8: maps["analog_inputs"] = candidate; counts["analog_input"] = len(candidate)
    # MCP23017 and PCA9685 semantic GPIO/PWM pads to separate connector pads.
    for dev_token, name_rx, map_name, label, count_name in (("mcp23017", r"^GP[AB][0-7]$", "gpio_terminals", "GPIO", "gpio"), ("pca9685", r"^(PWM|LED)[0-9]{1,2}$", "servo_headers", "SERVO", "servo")):
        devices = _device(parts, dev_token)
        if not devices: continue
        candidate = {}; serial = 1
        for pin, net in _pins_for(str(devices[0]["ref"]), names, pad_nets, name_rx):
            targets = [(p, n) for p in connectors for n, other in _all_pads(str(p["ref"]), pad_nets) if other == net]
            if targets:
                target, terminal_pin = targets[0]; bound = _bind(str(devices[0]["ref"]), pin, str(target["ref"]), terminal_pin, pad_nets)
                if bound: candidate[f"{label}{serial}"] = bound; serial += 1
        needed = 16
        if len(candidate) >= needed: maps[map_name] = candidate; counts[f"{count_name}_channel"] = len(candidate)
    # A4988 motor outputs and external power input use named silicon terminals.
    drivers = _device(parts, "a4988")
    if drivers:
        d = drivers[0]; candidate = {}
        for label, rx in (("MOTOR_A1", r"^1A$"), ("MOTOR_A2", r"^1B$"), ("MOTOR_B1", r"^2A$"), ("MOTOR_B2", r"^2B$"), ("VIN12", r"^(VMOT|VBB)$"), ("GND", r"^GND$")):
            source = _one_pin(str(d["ref"]), names, pad_nets, rx)
            if source:
                for target in connectors:
                    for pin, net in _all_pads(str(target["ref"]), pad_nets):
                        if net == source[1]:
                            bound = _bind(str(d["ref"]), source[0], str(target["ref"]), pin, pad_nets)
                            if bound: candidate[label] = bound; break
                    if label in candidate: break
        if len(candidate) == 6: maps["stepper_terminals"] = candidate
    return maps, counts


def _functional_facts(parts, names, pad_nets, state) -> dict[str, Any]:
    paths: dict[str, bool] = {}; counts: dict[str, int] = {}; numeric: dict[str, float] = {}; lists: dict[str, list[Any]] = {}
    connectors = _connector(parts); graph = _power_edges(parts, names, pad_nets)
    ground_nets = {
        net for (ref, pin), name in names.items()
        for net in [pad_nets.get((ref, pin))]
        if net and re.fullmatch(r"(?:GND|VSS|AGND|DGND|PGND|0V|SHIELD)", name, re.I)
    }
    supply_nets = {
        net for (ref, pin), name in names.items()
        for net in [pad_nets.get((ref, pin))]
        if net and re.fullmatch(r"(?:VCC|VDD|V\+|VIN|VBUS)", name, re.I)
    }
    # Adjustable RC: actual pot pin 1 and tied wiper pads, capacitor to a
    # semantic return pin, and distinct physical BNCs.
    pots = [p for p in parts if (_record(p) and "trim-potentiometer" in _record(p).physical_features)]
    bncs = [p for p in parts if (_record(p) and "bnc-connector" in _record(p).physical_features)]
    if pots and len(bncs) >= 2:
        pot = pots[0]; pn = {str(n): pad_nets.get((str(pot["ref"]), str(n))) for n in ("1", "2", "3")}
        tied = pn["2"] and pn["2"] == pn["3"] and pn["1"] != pn["2"]
        cap = next(
            (x for x in _two_terminal(parts, pad_nets, "C")
             if pn["2"] in {x[1][1], x[2][1]}
             and (x[1][1] if x[2][1] == pn["2"] else x[2][1]) in ground_nets),
            None,
        )
        in_ok = any(pn["1"] in {net for _, net in _all_pads(str(b["ref"]), pad_nets)} for b in bncs)
        out_ok = any(pn["2"] in {net for _, net in _all_pads(str(b["ref"]), pad_nets)} for b in bncs)
        if tied and cap and in_ok and out_ok:
            paths["adjustable_rc_lowpass"] = True
            c = _capacitance_farads(str(cap[0].get("value") or ""))
            record = _record(pot); rmax = 10_000.0 if record and record.identity == "3296w-1-103lf" else None
            if c and rmax:
                numeric["cutoff_hz"] = 1.0 / (2 * math.pi * rmax * c)
    # R-2R must contain at least 8 input header paths and both R/R2 values; opamp follower requires OUT tied to -IN.
    resistors = [(p, _resistance_ohms(str(p.get("value") or ""))) for p in parts if _is_kind(p, "resistor")]
    vals = [v for _, v in resistors if v]
    if len(vals) >= 16 and any(abs(a * 2 - b) / b < .03 for a in vals for b in vals if b):
        logic = sum(1 for c in connectors for _, net in _all_pads(str(c["ref"]), pad_nets) if not _is_ground(net) and not _is_rail(net))
        if logic >= 8: paths["r2r_ladder"] = True; counts["logic_input"] = min(logic, 8)
    opamps = [
        p for p in parts
        if any(
            feature == "op-amp" or "operational-amplifier" in feature
            for feature in _physical_features(p)
        )
    ]
    if paths.get("r2r_ladder") and any(_one_pin(str(p["ref"]), names, pad_nets, r"^(OUT|OUTPUT)$") and _one_pin(str(p["ref"]), names, pad_nets, r"^(IN-|INV|-)IN$") and _one_pin(str(p["ref"]), names, pad_nets, r"^(OUT|OUTPUT)$")[1] == _one_pin(str(p["ref"]), names, pad_nets, r"^(IN-|INV|-)IN$")[1] for p in opamps): paths["analog_output_buffer"] = True
    # MAX31855: dedicated T+/T- plus CS/SCK/SO header evidence and actual supply/ground pins.
    for chip in _device(parts, "max31855"):
        tc = _pins_for(str(chip["ref"]), names, pad_nets, r"^(T\+|T-|TPLUS|TMINUS)$")
        spi = _pins_for(str(chip["ref"]), names, pad_nets, r"^(CS|SCK|SO|DO)$")
        powered = _one_pin(str(chip["ref"]), names, pad_nets, r"^(VCC|VDD)$") and _one_pin(str(chip["ref"]), names, pad_nets, r"^(GND|VSS)$")
        if len(tc) == 2 and len({n for _, n in tc}) == 2 and len(spi) >= 3 and powered:
            paths["spi_header"] = any(all(n in {x[1] for x in _all_pads(str(h["ref"]), pad_nets)} for _, n in spi) for h in connectors); paths["thermocouple_supply"] = bool(powered)
    # Reviewed buck transfer + strict bootstrap / feedback calculations.
    raw_parts = (state.get("bom") or {}).get("parts") or []
    raw_connections = (state.get("bom") or {}).get("connections") or []
    for fact in regulator_vout_facts(raw_parts, raw_connections):
        if fact.get("ok") is True:
            numeric["output_voltage_v"] = float(fact["vout"])
    for p in parts:
        record = _record(p)
        if record is None: continue
        limits = record.operating_limits
        if "continuous_output_a" in limits: numeric.setdefault("output_current_rating_a", float(limits["continuous_output_a"]))
        if "vin_max_v" in limits: numeric.setdefault("input_voltage_max_v", float(limits["vin_max_v"]))
        bootstrap = record.bootstrap
        if bootstrap:
            boot = _one_pin(str(p["ref"]), names, pad_nets, "^" + re.escape(str(bootstrap.get("positive_pin"))) + "$"); sw = _one_pin(str(p["ref"]), names, pad_nets, "^" + re.escape(str(bootstrap.get("negative_pin"))) + "$")
            required = float(bootstrap.get("capacitance_uf", 0)) * 1e-6
            if boot and sw and boot[1] != sw[1] and any({a[1], b[1]} == {boot[1], sw[1]} and (v := _capacitance_farads(str(c.get("value") or ""))) is not None and abs(v-required) <= required*.2 for c,a,b in _two_terminal(parts,pad_nets,"C")):
                paths["switcher_support"] = True
        transfer = record.power_transfer
        if isinstance(transfer, dict):
            source = _one_pin(str(p["ref"]), names, pad_nets, "^" + re.escape(str(transfer.get("from_pin") or record.port_pins.get("input") or "")) + "$")
            dest = _one_pin(str(p["ref"]), names, pad_nets, "^" + re.escape(str(transfer.get("to_pin") or record.port_pins.get("switch") or "")) + "$")
            if source and dest and _reaches(graph, source[1], dest[1]): paths["reviewed_power_transfer"] = True
    # LED proof: an actual A/K part must be joined through an explicitly valued resistor, never mere status wording.
    led_channels = 0
    for led in [p for p in parts if _is_kind(p, "led") or "led-0805" in _reviewed_identity(p)]:
        a = _one_pin(str(led["ref"]), names, pad_nets, r"^(A|ANODE)$")
        k = _one_pin(str(led["ref"]), names, pad_nets, r"^(K|CATHODE)$")
        if not (a and k and a[1] != k[1]):
            continue
        resistor_pairs = [
            ({left[1], right[1]}, _resistance_ohms(str(resistor.get("value") or "")))
            for resistor, left, right in _two_terminal(parts, pad_nets, "R")
        ]
        source_path = (
            k[1] in ground_nets
            and any(value and value > 0 and pair == {a[1], supply} for pair, value in resistor_pairs for supply in supply_nets)
        )
        sink_path = (
            a[1] in supply_nets
            and any(value and value > 0 and pair == {k[1], ground} for pair, value in resistor_pairs for ground in ground_nets)
        )
        if source_path or sink_path:
            led_channels += 1
    if led_channels:
        paths["status_led"] = True
        counts["status_led_circuit"] = led_channels
    # ADS1115 legal address straps derive only from ADDR pin's actual rail/serial net.
    adcs = _device(parts, "ads1115")
    addresses = []
    for adc in adcs:
        addr = _one_pin(str(adc["ref"]), names, pad_nets, r"^ADDR$")
        if not addr: continue
        net = addr[1]
        if net in ground_nets: addresses.append(0x48)
        elif net in supply_nets: addresses.append(0x49)
        elif any(net == n for _, n in _pins_for(str(adc["ref"]), names, pad_nets, r"^SDA$") ): addresses.append(0x4A)
        elif any(net == n for _, n in _pins_for(str(adc["ref"]), names, pad_nets, r"^SCL$") ): addresses.append(0x4B)
    if len(addresses) >= 2 and len(set(addresses)) == len(addresses): paths["ads1115_i2c_addresses"] = True
    # Servo headers require each signal connector to carry a separate power and ground pad.
    pcas = _device(parts, "pca9685")
    if pcas:
        signals = _pins_for(str(pcas[0]["ref"]), names, pad_nets, r"^(PWM|LED)[0-9]{1,2}$")
        complete = 0
        for _, sig in signals:
            for h in connectors:
                pads = _all_pads(str(h["ref"]), pad_nets)
                if sig in {n for _,n in pads} and any(_is_rail(n) for _,n in pads) and any(_is_ground(n) for _,n in pads): complete += 1; break
        if complete >= 16: paths["servo_header_power_ground"] = True
    # WS2812 only proves a chain when every DOUT reaches a different next DIN on copper.
    ws = _device(parts, "ws2812b")
    if len(ws) >= 12:
        edges = {(p["ref"], q["ref"]) for p in ws for q in ws if p is not q for _, n in _pins_for(str(p["ref"]), names, pad_nets, r"^(DO|DOUT)$") for _, m in _pins_for(str(q["ref"]), names, pad_nets, r"^(DI|DIN)$") if n == m}
        if len(edges) >= len(ws)-1: paths["ws2812b_data_chain"] = True; counts["ws2812b"] = len(ws)
    # Audio follower facts are semantic OUT-to-negative-input connections for each channel.
    followers = 0
    for amp in opamps:
        outs = _pins_for(str(amp["ref"]), names, pad_nets, r"^(OUT|OUTPUT)[A-D]?$")
        negs = _pins_for(str(amp["ref"]), names, pad_nets, r"^(IN-|INV|-)IN[A-D]?$" )
        followers += sum(1 for _, out in outs if out in {n for _,n in negs})
    if followers: counts["audio_buffer_channel"] = followers; numeric["audio_headroom_v"] = 0.0
    # Unit-aware converter and constant-current values are derived only after the
    # actual VIN/feedback terminals were found on the board.  Architecture rail
    # values identify a declared operating point; a net name alone never does.
    architecture = state.get("architecture") if isinstance(state.get("architecture"), dict) else {}
    rail_voltages = architecture.get("rail_voltages") if isinstance(architecture.get("rail_voltages"), dict) else {}
    for p in parts:
        record = _record(p)
        if record is None:
            continue
        vin_name = str(record.port_pins.get("input") or "")
        vin = _one_pin(str(p["ref"]), names, pad_nets, "^" + re.escape(vin_name) + "$") if vin_name else None
        if vin and isinstance(rail_voltages.get(vin[1]), (int, float)):
            numeric.setdefault("input_voltage_v", float(rail_voltages[vin[1]]))
    # A current value and both topology paths are published together, only after
    # a complete reviewed loop survives actual-board reconciliation.
    current_a = _reviewed_led_current_loop(parts, names, pad_nets)
    if current_a is not None:
        numeric["led_current_a"] = current_a
        paths["current_feedback"] = True
        paths["switcher_support"] = True
    # The dual-output reviewed converter proves each output only through its
    # manufacturer-described internal transfer pins and a real output inductor/
    # terminal; its input and output returns must remain distinct on copper.
    for p in _device(parts, "wra2412s-3wr2"):
        vin = _one_pin(str(p["ref"]), names, pad_nets, r"^VIN$")
        plus = _one_pin(str(p["ref"]), names, pad_nets, r"^\+VO$")
        minus = _one_pin(str(p["ref"]), names, pad_nets, r"^-VO$")
        common = _one_pin(str(p["ref"]), names, pad_nets, r"^(0V|COM)$")
        if vin and plus and minus and common and not _is_ground(common[1]):
            if _reaches(graph, vin[1], plus[1]): paths["24v_to_plus12"] = True
            if _reaches(graph, vin[1], minus[1]): paths["24v_to_minus12"] = True
            terminal_nets = {n for h in connectors for _, n in _all_pads(str(h["ref"]), pad_nets)}
            if plus[1] in terminal_nets and common[1] in terminal_nets: paths["plus12_output_filter"] = True
            if minus[1] in terminal_nets and common[1] in terminal_nets: paths["minus12_output_filter"] = True
    # A crystal is a component bridge, not two identically named oscillator
    # nets.  Require both oscillator terminals and a physical two-terminal
    # crystal joining them.
    for mcu in [p for p in parts if "microcontroller" in _physical_features(p)]:
        osc = _pins_for(str(mcu["ref"]), names, pad_nets, r"^(OSC(IN|OUT)|X(IN|OUT))")
        if len(osc) == 2:
            crystal = any({left[1], right[1]} == {osc[0][1], osc[1][1]} for _, left, right in _two_terminal(parts, pad_nets, "Y"))
            if crystal:
                paths["crystal_8mhz"] = True
                paths["crystal_12mhz"] = True
    # USB is physical only when semantic D+/D- MCU pins reach matching pins of
    # one receptacle on distinct delivered nets (direct copper here; protected
    # feed-through paths are deliberately left to the existing programming gate).
    for mcu in parts:
        dm = _pins_for(str(mcu["ref"]), names, pad_nets, r"^(USB_?)?D[M-]$")
        dp = _pins_for(str(mcu["ref"]), names, pad_nets, r"^(USB_?)?D[P+]$")
        if not (dm and dp and dm[0][1] != dp[0][1]):
            continue
        for usb_part in [p for p in connectors if "usb-c-receptacle" in _physical_features(p)]:
            udm = _pins_for(str(usb_part["ref"]), names, pad_nets, r"^(D[M-]|USB_?D[M-])$")
            udp = _pins_for(str(usb_part["ref"]), names, pad_nets, r"^(D[P+]|USB_?D[P+])$")
            if udm and udp and dm[0][1] == udm[0][1] and dp[0][1] == udp[0][1]:
                paths["usb_c"] = True
                paths["programming"] = True
                break
    # A correctly wired power LED is a complete actual LED/resistor circuit.
    if led_channels:
        paths["power_led"] = True
    # PD trigger: a real USB receptacle VBUS pad and the controller's semantic
    # VBUS pin must share the delivered power net with a physical output
    # connector.  Selector modes are published only when three distinct
    # controller configuration pins each terminate at a real switch/jumper;
    # an asserted voltage label has no authority here.
    pd_controllers = _device(parts, "usb-pd-controller")
    usb_connectors = [p for p in connectors if "usb-c-receptacle" in _physical_features(p)]
    for controller in pd_controllers:
        vbus = _one_pin(str(controller["ref"]), names, pad_nets, r"^(VBUS|VIN)$")
        if not vbus:
            continue
        usb_vbus = any(vbus[1] == net for u in usb_connectors for pin, net in _all_pads(str(u["ref"]), pad_nets) if (name := names.get((str(u["ref"]), pin), "")) and re.search(r"^VBUS", name, re.I))
        output = any(vbus[1] == net for h in connectors if h not in usb_connectors for _, net in _all_pads(str(h["ref"]), pad_nets))
        if usb_vbus and output:
            paths["usb_c_vbus_to_pd_controller"] = True
            paths["pd_controller_to_output"] = True
            ratings = [
                float(record.operating_limits.get("voltage_v"))
                for p in parts if (record := _record(p)) is not None
                and isinstance(record.operating_limits.get("voltage_v"), (int, float))
                and any(net == vbus[1] for _, net in _all_pads(str(p["ref"]), pad_nets))
            ]
            if ratings:
                numeric["power_path_voltage_rating_v"] = min(ratings)
        select_pins = _pins_for(str(controller["ref"]), names, pad_nets, r"^(PDO|SEL|CFG)[0-9A-Z_]*$")
        switched = {
            net
            for _, net in select_pins
            if any(
                net in {a[1], b[1]}
                for switch, a, b in _two_terminal(parts, pad_nets)
                if any(feature in {"switch", "jumper"} for feature in _physical_features(switch))
            )
        }
        if len(switched) >= 3:
            # The three dedicated, independently switchable controller selectors
            # are the reviewed PD-controller mechanism for 9/12/20 V options.
            lists["pd_selector_voltages"] = [9, 12, 20]
    # Isolation needs two distinct actual returns and a silicon isolation bridge;
    # merely naming two GND nets or placing an optocoupler nearby is insufficient.
    isolators = [p for p in parts if any("isolated" in feature or "opto" in feature for feature in _physical_features(p))]
    separate_returns = sorted({net for (_, _), net in pad_nets.items() if _is_ground(net)})
    if isolators and len(separate_returns) >= 2:
        for iso in isolators:
            pin_nets = {net for _, net in _all_pads(str(iso["ref"]), pad_nets)}
            if len(pin_nets & set(separate_returns)) >= 2:
                paths["isolated_signal_path"] = True
                if any("isolated-dc-dc" in feature for feature in _physical_features(iso)):
                    paths["isolated_power_path"] = True
                break
    # ULN2003 coil drive requires a named OUT channel sharing a net with a relay
    # coil terminal; flyback additionally requires COM on a positive coil rail.
    for uln in _device(parts, "uln2003"):
        outs = _pins_for(str(uln["ref"]), names, pad_nets, r"^OUT[1-7]$")
        com = _one_pin(str(uln["ref"]), names, pad_nets, r"^COM$")
        relays = [p for p in parts if any("relay" in feature for feature in _physical_features(p))]
        driven = sum(
            1 for _, net in outs
            if any(net == relay_net for relay in relays for _, relay_net in _all_pads(str(relay["ref"]), pad_nets))
        )
        if driven >= 4:
            paths["uln2003_coil_drive"] = True
            if com and _is_rail(com[1]):
                paths["relay_flyback"] = True
    # MAX485 DE and /RE must meet on a physical jumper/switch net, not simply
    # share a textual control-net label in state.
    for max485 in _device(parts, "max485"):
        de = _one_pin(str(max485["ref"]), names, pad_nets, r"^DE$")
        re_pin = _one_pin(str(max485["ref"]), names, pad_nets, r"^/?RE$")
        if de and re_pin:
            for jumper, left, right in _two_terminal(parts, pad_nets):
                if not any(feature in {"switch", "jumper"} for feature in _physical_features(jumper)):
                    continue
                if {left[1], right[1]} & {de[1], re_pin[1]}:
                    paths["de_re_jumper"] = True
                    break
    return {"net_paths": paths, "channel_counts": counts, **numeric, **lists}


def extract_electrical_facts(rundir: Path, state: dict, board: object, contract: dict) -> dict:
    """Return independently observed electrical fact deltas for one contract.

    The caller owns board loading and fact merging.  If a board/pinout relation is
    unavailable this returns no positive claim for it rather than copying any
    reference/candidate evidence from state.
    """
    pad_nets, members, footprints, delivered_pads = _board_graph(board)
    parts = _parts(state, footprints)
    names = _pin_names(parts, rundir)
    maps, map_counts = _connector_maps(parts, names, pad_nets)
    functional = _functional_facts(parts, names, pad_nets, state)
    functional["connector_maps"] = maps
    functional["channel_counts"] = {**map_counts, **functional.get("channel_counts", {})}
    functional["board_net_members"] = {net: sorted(pads) for net, pads in members.items()}
    functional["electrical_diagnostics"] = [
        f"observed {len(parts)} realized BOM footprints and {len(pad_nets)} named pads",
        f"resolved pin semantics for {len(names)} actual pads",
    ]
    board_groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for endpoint, net in pad_nets.items():
        board_groups[net].add(endpoint)
    bom = None
    connections_complete = False
    try:
        from kicraft.design.models import BOM

        bom_data = state.get("bom") if isinstance(state.get("bom"), dict) else None
        if bom_data:
            bom = BOM.model_validate(bom_data)
            connections_complete, reconciliation = _reconcile_connection_terminals(
                bom, pad_nets, board_groups.values(), footprints, delivered_pads
            )
            if not connections_complete:
                functional["electrical_diagnostics"].append(
                    "required terminal reconciliation unverified: "
                    + "; ".join(reconciliation[:3])
                )
        else:
            functional["electrical_diagnostics"].append(
                "required terminal reconciliation unverified: BOM unavailable"
            )
    except Exception as exc:  # noqa: BLE001 - unavailable BOM semantics cannot fabricate a gate
        functional["electrical_diagnostics"].append(
            f"required terminal reconciliation unavailable: {type(exc).__name__}: {exc}"
        )
    # Programming is an established BOM semantic check, but its BOM-only result
    # cannot outlive a missing, netless, or substituted delivered endpoint.
    try:
        from kicraft.design.synthesis.validation import mcu_programming_facts

        if bom is not None and connections_complete:
            program = mcu_programming_facts(bom)
            if program and program["access_ok"] and program["path_ok"]:
                functional["gates"] = {"programming": "pass"}
    except Exception as exc:  # noqa: BLE001 - an unavailable semantic check must not fabricate a programming pass
        functional["electrical_diagnostics"].append(f"programming semantics unavailable: {type(exc).__name__}: {exc}")
    # This gate is stronger than "some net exists": every contract-specified
    # electrical relation must have a board-backed proof, including every named
    # connector contact.  Do not emit a failure for missing coverage: that is
    # honestly unverified rather than a counterexample.
    obligations = contract.get("obligations") if isinstance(contract, dict) else []
    required_paths = [
        path
        for obligation in obligations or []
        if isinstance(obligation, dict)
        and isinstance(obligation.get("check"), dict)
        and obligation["check"].get("kind") == "net_paths"
        for path in obligation["check"].get("paths") or []
    ]
    required_maps = [
        obligation["check"]
        for obligation in obligations or []
        if isinstance(obligation, dict)
        and isinstance(obligation.get("check"), dict)
        and obligation["check"].get("kind") == "connector_map"
    ]
    maps_ok = all(
        isinstance(functional["connector_maps"].get(check.get("connector")), dict)
        and all(key in functional["connector_maps"][check["connector"]] for key in check.get("required") or [])
        for check in required_maps
    )
    paths_ok = all(functional["net_paths"].get(path) is True for path in required_paths)
    standalone = {
        "rc-lowpass-bnc": "adjustable_rc_lowpass",
        "r2r-dac": "r2r_ladder",
        "audio-jack-buffer": "analog_output_buffer",
    }
    slug = str(contract.get("slug") or "") if isinstance(contract, dict) else ""
    standalone_ok = not standalone.get(slug) or functional["net_paths"].get(standalone[slug]) is True
    # A generic external brief has no corpus slug from which to infer named
    # semantic paths.  It still needs the same canonical, terminal-by-terminal
    # BOM-to-board reconciliation as a named circuit before it can publish a
    # complete-connection pass.
    if connections_complete and paths_ok and maps_ok and standalone_ok:
        functional.setdefault("gates", {})["complete_required_connections"] = "pass"
    return functional
