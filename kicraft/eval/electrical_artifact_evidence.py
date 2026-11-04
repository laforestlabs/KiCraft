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
from kicraft.eval.analog_artifact_evidence import (
    analog_conditioner_facts,
    temt6000_readout_facts,
)
from kicraft.eval.actuator_artifact_evidence import actuator_circuit_facts
from kicraft.eval.application_pin_evidence import reviewed_application_pins
from kicraft.eval.display_artifact_evidence import display_circuit_facts
from kicraft.eval.converter_current_evidence import (
    qualified_converter_power_stage_currents,
    qualified_module_output_net,
)
from kicraft.eval.data_logging_artifact_evidence import data_logging_hardware_facts
from kicraft.eval.dual_source_power_evidence import observe_dual_source_battery_power
from kicraft.eval.temperature_artifact_evidence import temperature_acquisition_facts


from kicraft.design.part_identity import physical_inventory_record, reviewed_inventory
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


def _reviewed_footprint_agrees(reviewed: str, recorded: str) -> bool:
    """Whether a recorded board footprint is the reviewed library pair.

    The library item name must agree exactly; the nickname must agree only when
    the board carries one, because a saved board can drop it.  A prefix or family
    match is never enough: the reviewed pair is the land pattern the record's
    symbol and contacts were reviewed against.
    """
    reviewed_library, _, reviewed_item = str(reviewed or "").partition(":")
    recorded_library, separator, recorded_item = str(recorded or "").partition(":")
    if not separator:  # the board carried no library nickname
        recorded_library, recorded_item = "", recorded_library
    if not reviewed_item or not recorded_item:
        return False
    return recorded_item == reviewed_item and (not recorded_library or recorded_library == reviewed_library)

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
        if not _reviewed_footprint_agrees(expected_footprint, footprints.get(ref, "")):
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
        and _reviewed_footprint_agrees(
            str(p.get("footprint") or ""), footprints[str(p.get("ref"))]
        )
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
    if symbol in {"Jumper:SolderJumper_2_Open", "Jumper:SolderJumper_2_Closed"} and str(
        part.get("footprint") or ""
    ).startswith("Jumper:SolderJumper-2_"):
        return "jumper"
    features = _physical_features(part)
    if "schottky-diode" in features:
        return "diode"
    if "power-inductor" in features:
        return "inductor"
    if "led" in features:
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


def _is_reviewed_jumper(part: dict[str, Any]) -> bool:
    """Recognize only a solder jumper or an exact reviewed shunt-header pair."""
    if _is_kind(part, "jumper") or "jumper" in _physical_features(part):
        return True
    if str(part.get("mpn") or "").strip():
        return False
    symbol = str(part.get("symbol") or "")
    footprint = str(part.get("footprint") or "")
    return any(
        candidate.symbol == symbol
        and candidate.footprint == footprint
        and "jumper" in candidate.physical_features
        for candidate in reviewed_inventory()
    )





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


def _record_port_pins(
    ref: str,
    record: Any,
    port: str,
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
) -> list[tuple[str, str]]:
    """Resolve a reviewed port to delivered pads.

    Recent reviewed records deliberately use manufacturer *pin numbers* for
    power devices.  Those are already KiCad pad identifiers, not semantic
    pin names to look up in a symbol.  Older records use the symbol spelling.
    Prefer the numeric/direct delivered pad, then fall back to either reviewed
    spelling or the semantic port name.  A caller which requires one terminal
    must still reject an ambiguous result with ``_one_port_pin``.
    """
    declared = str(getattr(record, "port_pins", {}).get(port) or "")
    direct = (ref, declared)
    if declared and direct in pad_nets:
        return [(declared, pad_nets[direct])]
    wanted = {declared.casefold(), port.casefold()} - {""}
    return [
        (pin, net)
        for (candidate, pin), name in names.items()
        if candidate == ref
        and (ref, pin) in pad_nets
        and _normalized_pin_name(name).casefold() in wanted
        for net in [pad_nets[(ref, pin)]]
    ]


def _one_port_pin(
    ref: str,
    record: Any,
    port: str,
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
) -> tuple[str, str] | None:
    found = _record_port_pins(ref, record, port, names, pad_nets)
    return found[0] if len(found) == 1 else None


def _all_reviewed_pins(
    ref: str,
    record: Any,
    key: str,
    fallback_port: str,
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
) -> list[tuple[str, str]]:
    """Resolve a reviewed repeated-pin bus and require every listed pad."""
    support = getattr(record, "support_network", {}) or {}
    declared = support.get(key)
    if not isinstance(declared, (list, tuple)):
        return _record_port_pins(ref, record, fallback_port, names, pad_nets)
    resolved = [
        (str(pin), pad_nets[(ref, str(pin))])
        for pin in declared
        if (ref, str(pin)) in pad_nets
    ]
    return resolved if len(resolved) == len(declared) else []


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
            or record.family in {"pin-header", "binding-post", "usb-c-receptacle"}
            or record.family.endswith("-receptacle")
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
    """Return conductive passive paths, including bounded reviewed trim topology."""
    graph: dict[str, set[str]] = defaultdict(set)
    for p in parts:
        # A direct conductive primitive is established by its exact symbol, not
        # by the human-chosen reference prefix. Capacitors are never power-path
        # edges. A 3296W used as a rheostat has its wiper and CW end tied, so
        # its remaining two delivered nets are a real (variable) conductive
        # path; treating it as absent would hide external back-powering.
        if _symbol_kind(p) in {"resistor", "inductor"}:
            pins = _all_pads(str(p["ref"]), pad_nets)
            if len(pins) == 2 and pins[0][1] != pins[1][1]:
                graph[pins[0][1]].add(pins[1][1])
                graph[pins[1][1]].add(pins[0][1])
            continue
        record = _record(p)
        if record is None or record.identity != "3296w-1-103lf":
            continue
        ref = str(p["ref"])
        ccw = pad_nets.get((ref, "1"))
        wiper = pad_nets.get((ref, "2"))
        cw = pad_nets.get((ref, "3"))
        if not ccw or not wiper or wiper != cw or ccw == wiper:
            continue
        graph[ccw].add(wiper)
        graph[wiper].add(ccw)
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


def _usb_data_graph(parts, pad_nets) -> dict[str, set[str]]:
    """Join only measured low-value resistors and reviewed USB ESD paths."""
    graph: dict[str, set[str]] = defaultdict(set)
    for resistor, left, right in _two_terminal(parts, pad_nets, "R"):
        resistance = _resistance_ohms(str(resistor.get("value") or ""))
        if resistance is not None and 0 <= resistance <= 100:
            graph[left[1]].add(right[1])
            graph[right[1]].add(left[1])
    for esd in parts:
        record = _record(esd)
        if record is None:
            continue
        for first, second in (record.support_network or {}).get("usb_feed_through_pairs", ()):
            left = pad_nets.get((str(esd["ref"]), first))
            right = pad_nets.get((str(esd["ref"]), second))
            if left and right:
                graph[left].add(right)
                graph[right].add(left)
    return graph


def _power_edges(
    parts: list[dict[str, Any]], names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str], *, nonconverting_only: bool = False,
) -> dict[str, set[str]]:
    # Feedback, UVLO and gate-drive resistors are not power conductors.  Only
    # reviewed power-transfer parts and explicitly zero-ohm links may bridge
    # rails in this graph.
    graph: dict[str, set[str]] = defaultdict(set)
    for resistor, left, right in _two_terminal(parts, pad_nets, "R"):
        value = _resistance_ohms(str(resistor.get("value") or ""))
        if value is not None and value <= 0.1:
            graph[left[1]].add(right[1]); graph[right[1]].add(left[1])
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
            if nonconverting_only and path.get("preserves_nominal_voltage") is not True:
                continue
            ref = str(p["ref"])
            from_pin, to_pin = str(path.get("from_pin") or ""), str(path.get("to_pin") or "")
            source = [(from_pin, pad_nets[(ref, from_pin)])] if (ref, from_pin) in pad_nets else _record_port_pins(ref, record, "input", names, pad_nets)
            target = [(to_pin, pad_nets[(ref, to_pin)])] if (ref, to_pin) in pad_nets else _record_port_pins(ref, record, "output", names, pad_nets)
            if len(source) != 1 or len(target) != 1 or source[0][1] == target[0][1]:
                continue
            # One connected member cannot stand in for a broken repeated current
            # bus.  Require every reviewed bus pad before publishing its transfer.
            support = record.support_network or {}
            buses = (
                ("source_pins", "source", source[0][1])
                if "source_pins" in support else ("input_pins", "input", source[0][1]),
                ("drain_pins", "drain", target[0][1])
                if "drain_pins" in support else ("output_pins", "output", target[0][1]),
            )
            broken_bus = False
            for key, port, expected_net in buses:
                bus = _all_reviewed_pins(ref, record, key, port, names, pad_nets)
                if (key in support and not bus) or any(net != expected_net for _, net in bus):
                    broken_bus = True
                    break
            if broken_bus:
                continue
            graph[source[0][1]].add(target[0][1])
            graph[target[0][1]].add(source[0][1])
    return graph


def _two_terminal(parts: Iterable[dict[str, Any]], pad_nets: dict[tuple[str, str], str], kind: str | None = None):
    aliases = {"R": "resistor", "C": "capacitor", "L": "inductor", "Y": "crystal"}
    wanted = aliases.get(kind or "", kind)
    for p in parts:
        features = _physical_features(p)
        reviewed_inductor = wanted == "inductor" and any(
            feature.endswith("inductor") for feature in features
        )
        if wanted and _symbol_kind(p) != wanted and wanted not in features and not reviewed_inductor:
            continue
        pads = _all_pads(str(p["ref"]), pad_nets)
        if len(pads) == 2 and pads[0][1] != pads[1][1]:
            yield p, pads[0], pads[1]


def _board_connections(pad_nets: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    """Rebuild validator-shaped net groups exclusively from delivered pads."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (ref, pin), net in pad_nets.items():
        if net:
            grouped[net].append({"ref": ref, "pin": pin})
    return [
        {"net_name": net, "endpoints": sorted(endpoints, key=lambda endpoint: (endpoint["ref"], endpoint["pin"]))}
        for net, endpoints in sorted(grouped.items())
    ]


_ADJUSTABLE_VREF = {
    "lm2577t-adj/nopb": 1.230,
    "tps54331ddar": 0.800,
    "tps61088rhlr": 1.204,
    "tps552892ryqr": 1.200,
    "tps61175pwpr": 1.229,
    # TI TLV62569 SLVSDG1C §6.5: nominal FB regulation is 0.6 V.
    "tlv62569dbvr": 0.600,
}


def _resistor_path_value(
    start: str, end: str, parts: list[dict[str, Any]], pad_nets: dict[tuple[str, str], str], maximum: int,
) -> list[tuple[float, tuple[str, ...]]]:
    """Return unambiguous, short, series-resistor paths in the delivered board."""
    edges: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for resistor, left, right in _two_terminal(parts, pad_nets, "R"):
        value = _resistance_ohms(str(resistor.get("value") or ""))
        if value is not None and value > 0:
            edges[left[1]].append((right[1], value, str(resistor["ref"])))
            edges[right[1]].append((left[1], value, str(resistor["ref"])))
    found: list[tuple[float, tuple[str, ...]]] = []
    todo = [(start, 0.0, ())]
    while todo:
        net, resistance, refs = todo.pop()
        if len(refs) > maximum:
            continue
        if net == end and refs:
            found.append((resistance, refs))
            continue
        for nxt, value, ref in edges.get(net, ()):
            if ref not in refs:
                todo.append((nxt, resistance + value, (*refs, ref)))
    return found


def _divider_voltage(
    converter: dict[str, Any],
    record: Any,
    output_net: str,
    ground_net: str,
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
) -> float | None:
    """Calculate a reviewed divider, accepting a documented series top arm."""
    ref = str(converter["ref"])
    feedback = _one_port_pin(ref, record, "feedback", names, pad_nets)
    vref = _ADJUSTABLE_VREF.get(str(record.identity))
    if feedback is None or vref is None or feedback[1] in {output_net, ground_net}:
        return None
    tops = _resistor_path_value(output_net, feedback[1], parts, pad_nets, maximum=2)
    bottoms = _resistor_path_value(feedback[1], ground_net, parts, pad_nets, maximum=1)
    if len(tops) != 1 or len(bottoms) != 1 or set(tops[0][1]) & set(bottoms[0][1]):
        return None
    return vref * (1.0 + tops[0][0] / bottoms[0][0])


def _converter_ground(
    converter: dict[str, Any], record: Any, names: dict[tuple[str, str], str], pad_nets: dict[tuple[str, str], str],
) -> str | None:
    ref = str(converter["ref"])
    candidates: list[tuple[str, str]] = []
    for port in ("ground", "analog_ground", "power_ground"):
        candidates.extend(_record_port_pins(ref, record, port, names, pad_nets))
    nets = {net for _, net in candidates}
    return next(iter(nets)) if len(nets) == 1 else None






def _exact_reviewed_part(part: dict[str, Any], mpn: object) -> Any | None:
    """Return a delivered part only when its reviewed identity is exact."""
    record = _record(part)
    expected = str(mpn or "").casefold()
    if record is None or not expected or str(record.identity).casefold() != expected:
        return None
    return record


def _rated_capacitor_between(
    parts: list[dict[str, Any]],
    pad_nets: dict[tuple[str, str], str],
    left_net: str,
    right_net: str,
    requirement: dict[str, Any],
) -> bool:
    """Prove an exact reviewed capacitor and its published value/rating on two pads."""
    required_f = float(requirement.get("capacitance_uf") or 0) * 1e-6
    required_v = float(requirement.get("voltage_min_v") or 0)
    for capacitor, left, right in _two_terminal(parts, pad_nets, "C"):
        record = _exact_reviewed_part(capacitor, requirement.get("mpn"))
        if (
            record is not None
            and {left[1], right[1]} == {left_net, right_net}
            and (
                capacitance := record.operating_limits.get("capacitance_f")
                or float(record.operating_limits.get("capacitance_uf") or 0) * 1e-6
            ) >= required_f
            and float(
                record.operating_limits.get("voltage_max_v")
                or record.operating_limits.get("voltage_v")
                or 0
            ) >= required_v
        ):
            return True
    return False


def _rated_capacitor_feeding(
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
    input_net: str,
    ground_net: str,
    requirement: dict[str, Any],
) -> bool:
    """Prove a rated input bulk capacitor feeds the converter through power paths."""
    required_f = float(requirement.get("capacitance_uf") or 0) * 1e-6
    required_v = float(requirement.get("voltage_min_v") or 0)
    graph = _power_edges(parts, names, pad_nets)
    for capacitor, left, right in _two_terminal(parts, pad_nets, "C"):
        record = _exact_reviewed_part(capacitor, requirement.get("mpn"))
        if record is None or ground_net not in {left[1], right[1]}:
            continue
        rail = right[1] if left[1] == ground_net else left[1]
        if not (
            _reaches(graph, rail, input_net)
            and (
                capacitance := record.operating_limits.get("capacitance_f")
                or float(record.operating_limits.get("capacitance_uf") or 0)
                * 1e-6
            ) >= required_f
            and float(
                record.operating_limits.get("voltage_max_v")
                or record.operating_limits.get("voltage_v")
                or 0
            ) >= required_v
        ):
            continue
        return True
    return False


def _rated_inductor_between(
    parts: list[dict[str, Any]],
    pad_nets: dict[tuple[str, str], str],
    left_net: str,
    right_net: str,
    requirement: dict[str, Any],
) -> bool:
    """Prove the exact reviewed inductor and its published component limits."""
    required_h = float(requirement.get("inductance_uh") or 0) * 1e-6
    for inductor, left, right in _two_terminal(parts, pad_nets, "L"):
        record = _exact_reviewed_part(inductor, requirement.get("mpn"))
        if record is None or {left[1], right[1]} != {left_net, right_net}:
            continue
        limits = record.operating_limits
        if float(limits.get("inductance_h") or float(limits.get("inductance_uh") or 0) * 1e-6) < required_h:
            continue
        if all(
            float(limits.get(key) or 0) >= float(value)
            for key, value in requirement.items()
            if key.endswith("_a")
        ):
            return True
    return False


def _lm2577_support_ok(
    converter: dict[str, Any],
    record: Any,
    input_net: str,
    output_net: str,
    ground_net: str,
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
) -> bool:
    """Prove every manufacturer-specified LM2577 support branch on delivered pads."""
    requirements = (record.support_network or {}).get("artifact_requirements")
    if not isinstance(requirements, dict):
        return False
    input_bulk = requirements.get("input_bulk")
    inductor = requirements.get("inductor")
    rectifier = requirements.get("rectifier")
    output_bulk = requirements.get("output_bulk")
    compensation = requirements.get("compensation")
    if not all(
        isinstance(item, dict)
        for item in (input_bulk, inductor, rectifier, output_bulk, compensation)
    ) or not (
        input_bulk.get("converter_port") == "input"
        and input_bulk.get("return_port") == "ground"
        and inductor.get("input_relation") is True
        and inductor.get("switch_port") == "switch"
        and rectifier.get("anode_port") == "switch"
        and rectifier.get("cathode_relation") == "regulated_output"
        and output_bulk.get("regulated_output_relation") is True
        and compensation.get("converter_port") == "compensation"
        and compensation.get("return_port") == "ground"
    ):
        return False
    ref = str(converter["ref"])
    switch = _one_port_pin(ref, record, "switch", names, pad_nets)
    compensation_pin = _one_port_pin(ref, record, "compensation", names, pad_nets)
    if switch is None or compensation_pin is None:
        return False
    if not (
        _rated_capacitor_feeding(
            parts, names, pad_nets, input_net, ground_net, input_bulk
        )
        and _rated_inductor_between(parts, pad_nets, input_net, switch[1], inductor)
        and _rated_capacitor_between(parts, pad_nets, output_net, ground_net, output_bulk)
    ):
        return False
    rectifier_ok = any(
        _exact_reviewed_part(diode, rectifier.get("mpn")) is not None
        and (anode := _one_pin(str(diode["ref"]), names, pad_nets, r"^(A|ANODE)$")) is not None
        and (cathode := _one_pin(str(diode["ref"]), names, pad_nets, r"^(K|CATHODE)$")) is not None
        and anode[1] == switch[1]
        and cathode[1] == output_net
        and float(
            _exact_reviewed_part(diode, rectifier.get("mpn")).operating_limits.get(
                "forward_current_a"
            )
            or 0
        ) >= float(rectifier.get("forward_current_min_a") or 0)
        and float(
            _exact_reviewed_part(diode, rectifier.get("mpn")).operating_limits.get(
                "reverse_voltage_v"
            )
            or 0
        ) >= float(rectifier.get("reverse_voltage_min_v") or 0)
        for diode in parts
        if _is_kind(diode, "diode")
    )
    if not rectifier_ok:
        return False
    resistance = float(compensation.get("series_resistance_ohm") or 0)
    capacitance = float(compensation.get("capacitance_uf") or 0) * 1e-6
    capacitor_mpn = compensation.get("capacitor_mpn")
    if resistance <= 0 or capacitance <= 0 or not capacitor_mpn:
        return False
    for resistor, left, right in _two_terminal(parts, pad_nets, "R"):
        value = _resistance_ohms(str(resistor.get("value") or ""))
        if value is None or abs(value - resistance) > resistance * 0.01:
            continue
        if compensation_pin[1] not in {left[1], right[1]}:
            continue
        middle = right[1] if left[1] == compensation_pin[1] else left[1]
        if any(
            (cap_record := _exact_reviewed_part(capacitor, capacitor_mpn))
            is not None
            and {cap_left[1], cap_right[1]} == {middle, ground_net}
            and (
                cap_f := cap_record.operating_limits.get("capacitance_f")
                or float(cap_record.operating_limits.get("capacitance_uf") or 0)
                * 1e-6
            ) >= capacitance
            and float(
                cap_record.operating_limits.get("voltage_max_v")
                or cap_record.operating_limits.get("voltage_v")
                or 0
            ) > 0
            for capacitor, cap_left, cap_right in _two_terminal(parts, pad_nets, "C")
        ):
            return True
    return False
def _tps61175_boost_support_ok(converter, input_net, output_net, ground_net, parts, names, pad_nets):
    """Observe the finite current-limited USB boost population, not an IC rating."""
    ref = str(converter["ref"])
    net = lambda pin: pad_nets.get((ref, str(pin)))
    switch = net("1")
    if not switch or net("2") != switch or net("3") != input_net or net("4") != input_net:
        return False
    if any(net(pin) != ground_net for pin in ("6", "7", "11", "12", "13", "14", "15")):
        return False
    ss, comp, frequency = net("5"), net("8"), net("10")
    if not all((ss, comp, frequency)) or len({input_net, output_net, ground_net, switch, ss, comp, frequency}) != 7:
        return False
    resistors = list(_two_terminal(parts, pad_nets, "R"))
    capacitors = list(_two_terminal(parts, pad_nets, "C"))
    def resistor_between(left_net, right_net, value):
        return any(
            {left[1], right[1]} == {left_net, right_net}
            and (actual := _resistance_ohms(str(part.get("value") or ""))) is not None
            and math.isclose(actual, value, rel_tol=0.01)
            for part, left, right in resistors
        )
    def capacitor_between(left_net, right_net, value):
        return any(
            {left[1], right[1]} == {left_net, right_net}
            and (actual := _capacitance_farads(str(part.get("value") or ""))) is not None
            and math.isclose(actual, value, rel_tol=0.01)
            for part, left, right in capacitors
        )
    if not capacitor_between(ss, ground_net, 2.2e-6) or not resistor_between(frequency, ground_net, 80_000.0):
        return False
    compensation = any(
        comp in {left[1], right[1]}
        and (actual := _resistance_ohms(str(part.get("value") or ""))) is not None
        and math.isclose(actual, 1000.0, rel_tol=0.01)
        and (other := right[1] if left[1] == comp else left[1]) not in {comp, input_net, output_net, ground_net, switch, ss, frequency}
        and capacitor_between(other, ground_net, 330e-9)
        for part, left, right in resistors
    )
    if not compensation:
        return False
    if not _rated_inductor_between(parts, pad_nets, input_net, switch,
                                  {"mpn": "fxl0630-150-m", "inductance_uh": 15.0}):
        return False
    if not any(
        (diode_record := _exact_reviewed_part(part, "b360a")) is not None
        and (anode := _one_port_pin(str(part["ref"]), diode_record, "anode", names, pad_nets)) is not None
        and (cathode := _one_port_pin(str(part["ref"]), diode_record, "cathode", names, pad_nets)) is not None
        and anode[1] == switch and cathode[1] == output_net
        for part in parts
    ):
        return False
    for rail, count in ((input_net, 2), (output_net, 4)):
        refs = {
            str(part["ref"]) for part, left, right in capacitors
            if {left[1], right[1]} == {rail, ground_net}
            and _exact_reviewed_part(part, "grm31cr71h475ka12l") is not None
        }
        if len(refs) != count:
            return False
    # The USB operating point depends on its actual current limiter and ILIM,
    # not on an invented source-current capability or a soft-start guarantee.
    for limiter in parts:
        if _exact_reviewed_part(limiter, "tps2553dbvr") is None:
            continue
        lr = str(limiter["ref"])
        ln = lambda pin: pad_nets.get((lr, str(pin)))
        protected, ilim = ln("1"), ln("5")
        if (
            protected and protected not in {input_net, output_net, ground_net, switch}
            and ln("3") == protected and ln("2") == ground_net and ln("6") == input_net
            and ilim and ilim not in {protected, input_net, output_net, ground_net, switch}
            and resistor_between(ilim, ground_net, 61_900.0)
            and capacitor_between(protected, ground_net, 100e-9)
        ):
            return True
    return False


def _converter_support_ok(
    converter: dict[str, Any],
    record: Any,
    input_net: str,
    output_net: str,
    ground_net: str,
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
) -> bool:
    """Check only manufacturer-described support whose delivered pads prove it."""
    if record.identity == "tps61175pwpr":
        return _tps61175_boost_support_ok(
            converter, input_net, output_net, ground_net, parts, names, pad_nets
        )
    if record.identity == "lm2577t-adj/nopb":
        return _lm2577_support_ok(
            converter,
            record,
            input_net,
            output_net,
            ground_net,
            parts,
            names,
            pad_nets,
        )
    # The published TPS61088/TPS552892 artifact contracts explicitly leave their
    # delivery-MPN effective output capacitance unverified.  Do not treat nominal
    # capacitance, topology, or the IC's switch limit as a converted-output pass.
    if record.identity in {"tps61088rhlr", "tps552892ryqr"}:
        return False
    capacitors = list(_two_terminal(parts, pad_nets, "C"))
    if not any(
        {left[1], right[1]} == {input_net, ground_net}
        for _, left, right in capacitors
    ):
        return False
    bootstrap = record.bootstrap
    if bootstrap:
        ref = str(converter["ref"])
        positive = str(bootstrap.get("positive_pin") or "")
        negative = str(bootstrap.get("negative_pin") or "")
        boot = (
            (positive, pad_nets[(ref, positive)])
            if (ref, positive) in pad_nets
            else _one_pin(ref, names, pad_nets, "^" + re.escape(positive) + "$")
        )
        sw = (
            (negative, pad_nets[(ref, negative)])
            if (ref, negative) in pad_nets
            else _one_pin(ref, names, pad_nets, "^" + re.escape(negative) + "$")
        )
        required = float(bootstrap.get("capacitance_uf") or 0) * 1e-6
        if not boot or not sw or required <= 0 or not any(
            {left[1], right[1]} == {boot[1], sw[1]}
            and (
                value := _capacitance_farads(str(capacitor.get("value") or ""))
            ) is not None
            and abs(value - required) <= required * 0.2
            for capacitor, left, right in capacitors
        ):
            return False
    return True
def _switcher_output_net(
    converter: dict[str, Any],
    record: Any,
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
) -> str | None:
    """Prove a complete manufacturer-shaped converter, including internal rails."""
    ref = str(converter["ref"])
    input_pin = _one_port_pin(ref, record, "input", names, pad_nets)
    ground_net = _converter_ground(converter, record, names, pad_nets)
    if input_pin is None or ground_net is None or input_pin[1] == ground_net:
        return None
    identity = str(record.identity)
    if identity == "k7803m-1000r3":
        return qualified_module_output_net(converter, parts, pad_nets)
    four_switch = identity == "tps552892ryqr"
    if four_switch:
        switches = [
            *_record_port_pins(ref, record, "buck_switch", names, pad_nets),
            *_record_port_pins(ref, record, "boost_switch", names, pad_nets),
        ]
    else:
        switches = _all_reviewed_pins(ref, record, "switch_pins", "switch", names, pad_nets)
        if not switches:
            switches = _record_port_pins(ref, record, "switch", names, pad_nets)
    switch_nets = {net for _, net in switches}
    if (four_switch and len(switch_nets) != 2) or (not four_switch and len(switch_nets) != 1):
        return None
    switch_net = next(iter(switch_nets))
    output = _one_port_pin(ref, record, "output", names, pad_nets)
    is_boost = "boost" in record.family
    if output is not None:
        output_net = output[1]
    elif is_boost:
        # LM2577's switched node reaches the output only through the reviewed
        # catch diode; its anode/cathode orientation is material.
        diode_outputs = [
            cathode[1]
            for diode in parts if _is_kind(diode, "diode")
            if (anode := _one_pin(str(diode["ref"]), names, pad_nets, r"^(A|ANODE)$"))
            and (cathode := _one_pin(str(diode["ref"]), names, pad_nets, r"^(K|CATHODE)$"))
            and anode[1] == switch_net and cathode[1] != switch_net
        ]
        output_net = diode_outputs[0] if len(set(diode_outputs)) == 1 else ""
    else:
        inductors = [
            right[1] if left[1] == switch_net else left[1]
            for _, left, right in _two_terminal(parts, pad_nets, "L")
            if switch_net in {left[1], right[1]}
        ]
        output_net = inductors[0] if len(set(inductors)) == 1 else ""
    if not output_net:
        return None
    if four_switch:
        topology_distinct = len({input_pin[1], ground_net, output_net, *switch_nets}) == 5
        inductor_ok = any(
            {left[1], right[1]} == switch_nets
            for _, left, right in _two_terminal(parts, pad_nets, "L")
        )
    elif is_boost:
        topology_distinct = len({input_pin[1], switch_net, ground_net, output_net}) == 4
        inductor_ok = any(
            {left[1], right[1]} == {input_pin[1], switch_net}
            for _, left, right in _two_terminal(parts, pad_nets, "L")
        )
    else:
        topology_distinct = len({input_pin[1], switch_net, ground_net, output_net}) == 4
        inductor_ok = any(
            {left[1], right[1]} == {switch_net, output_net}
            for _, left, right in _two_terminal(parts, pad_nets, "L")
        )
    if not topology_distinct:
        return None
    output_cap_ok = any(
        {left[1], right[1]} == {output_net, ground_net}
        for _, left, right in _two_terminal(parts, pad_nets, "C")
    )
    if not (inductor_ok and output_cap_ok and _converter_support_ok(
        converter,
        record,
        input_pin[1],
        output_net,
        ground_net,
        parts,
        names,
        pad_nets,
    )):
        return None
    # Adjustable parts are incomplete without a measured feedback topology.
    if identity in _ADJUSTABLE_VREF and _divider_voltage(
        converter, record, output_net, ground_net, parts, names, pad_nets
    ) is None:
        return None
    if identity not in _ADJUSTABLE_VREF and "feedback" in record.port_pins:
        feedback = _one_port_pin(ref, record, "feedback", names, pad_nets)
        if feedback is None or feedback[1] != output_net:
            return None
    return output_net


def _observed_power_outputs(parts, names, pad_nets, state, source_outputs):
    """Resolve physically supplied converters and linear chains, not islands."""
    observed = list(source_outputs)
    pending = []
    for converter in parts:
        record = _record(converter)
        if record is None:
            continue
        output_net = _switcher_output_net(converter, record, parts, names, pad_nets)
        if output_net is None:
            continue
        ground_net = _converter_ground(converter, record, names, pad_nets)
        voltage = record.operating_limits.get("output_voltage_v")
        if str(record.identity) in _ADJUSTABLE_VREF and ground_net is not None:
            voltage = _divider_voltage(converter, record, output_net, ground_net, parts, names, pad_nets)
        if not isinstance(voltage, (int, float)) or isinstance(voltage, bool) or not math.isfinite(voltage) or voltage <= 0:
            continue
        current = record.operating_limits.get("continuous_output_a")
        if not isinstance(current, (int, float)):
            current = (record.support_network or {}).get("reference_boost", {}).get("continuous_output_max_a")
        pending.append((record, {
            "ref": str(converter["ref"]),
            "input_net": _one_port_pin(str(converter["ref"]), record, "input", names, pad_nets)[1],
            "output_net": output_net, "ground_net": ground_net, "voltage": float(voltage),
            "current_a": float(current) if isinstance(current, (int, float)) else None,
            "converting": True,
        }))
    while True:
        count = len(observed)
        remaining = []
        for record, output in pending:
            source = _supply_voltage_observed(
                output["input_net"], output["ground_net"], parts, names, pad_nets, state, observed,
            )
            limits = record.operating_limits
            minimum = limits.get("vin_min_v", limits.get("input_voltage_min_v", 0.0))
            maximum = limits.get("vin_max_v", limits.get("input_voltage_max_v", math.inf))
            exclusive_maximum = record.support_network.get("input_voltage_max_exclusive_v", math.inf)
            if source is not None and minimum <= source <= maximum and source < exclusive_maximum:
                observed.append(output)
            else:
                remaining.append((record, output))
        pending = remaining
        known = {output["ref"] for output in observed}
        observed.extend(
            {**output, "converting": True}
            for output in _observed_linear_outputs(parts, names, pad_nets, state, observed)
            if output["ref"] not in known
        )
        if len(observed) == count:
            return observed


def _exposed_power_outputs(outputs, parts, connectors, names, pad_nets):
    """Bind qualified outputs to physical contacts without crossing conversion."""
    graph = _power_edges(parts, names, pad_nets, nonconverting_only=True)
    return [
        output for output in outputs
        if any(
            _reaches(graph, output["output_net"], contact)
            and output["ground_net"] in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
            for connector in connectors
            for _, contact in _all_pads(str(connector["ref"]), pad_nets)
        )
    ]





def _power_input_pairs(
    parts: list[dict[str, Any]], names: dict[tuple[str, str], str], pad_nets: dict[tuple[str, str], str],
) -> list[tuple[str, str]]:
    """Physical input/return pairs of realized power stages and analog loads."""
    pairs: list[tuple[str, str]] = []
    for part in parts:
        record = _record(part)
        if record is None:
            continue
        if "dual-operational-amplifier" in record.physical_features:
            positive = _one_port_pin(str(part["ref"]), record, "vplus", names, pad_nets)
            negative = _one_port_pin(str(part["ref"]), record, "vminus", names, pad_nets)
            if positive and negative and positive[1] != negative[1]:
                pairs.append((positive[1], negative[1]))
            continue
        transfer = record.power_transfer
        power_device = bool(transfer) or any(
            token in record.family
            for token in ("converter", "regulator", "power-switch")
        ) or "current-limited-power-switch" in record.physical_features
        if not power_device:
            continue
        input_pin = _one_port_pin(str(part["ref"]), record, "input", names, pad_nets)
        ground = _converter_ground(part, record, names, pad_nets)
        if input_pin and ground and input_pin[1] != ground:
            pairs.append((input_pin[1], ground))
    return pairs


def _tps26600_return_isolated(
    parts: list[dict[str, Any]], names: dict[tuple[str, str], str], pad_nets: dict[tuple[str, str], str],
) -> bool:
    """TPS26600's RTN/PowerPAD is a control return, never load/source GND."""
    for part in _device(parts, "tps26600pwpr"):
        record = _record(part)
        if record is None:
            continue
        system = _one_port_pin(str(part["ref"]), record, "ground", names, pad_nets)
        control = _one_port_pin(str(part["ref"]), record, "control_return", names, pad_nets)
        thermal = _one_port_pin(str(part["ref"]), record, "thermal_return", names, pad_nets)
        if not system or not control or not thermal or control[1] == system[1] or thermal[1] != control[1]:
            return False
    return True


def _tps26600_support_observed(parts, names, pad_nets) -> bool:
    """Require the delivered TPS26600 protection divider and basic support."""
    for part in _device(parts, "tps26600pwpr"):
        record = _record(part)
        if record is None:
            continue
        ref = str(part["ref"])
        vin = _all_reviewed_pins(ref, record, "input_pins", "input", names, pad_nets)
        vout = _all_reviewed_pins(ref, record, "output_pins", "output", names, pad_nets)
        rtn = _one_port_pin(ref, record, "control_return", names, pad_nets)
        gnd = _one_port_pin(ref, record, "ground", names, pad_nets)
        uvlo = _one_port_pin(ref, record, "undervoltage", names, pad_nets)
        ovp = _one_port_pin(ref, record, "overvoltage", names, pad_nets)
        limit = _one_port_pin(ref, record, "limit", names, pad_nets)
        if not vin or not vout or not rtn or not gnd or not uvlo or not ovp or not limit:
            return False
        vin_nets, out_nets = {net for _, net in vin}, {net for _, net in vout}
        if len(vin_nets) != 1 or len(out_nets) != 1 or rtn[1] == gnd[1]:
            return False
        vin_net, out_net = next(iter(vin_nets)), next(iter(out_nets))
        chain_ok = all((
            len(_resistor_path_value(vin_net, uvlo[1], parts, pad_nets, maximum=3)) == 1,
            len(_resistor_path_value(uvlo[1], ovp[1], parts, pad_nets, maximum=3)) == 1,
            len(_resistor_path_value(ovp[1], rtn[1], parts, pad_nets, maximum=3)) == 1,
        ))
        ilim = _resistor_path_value(limit[1], rtn[1], parts, pad_nets, maximum=1)
        ilim_ok = len(ilim) == 1 and record.support_network["ilim_resistor_ohm_min"] <= ilim[0][0] <= record.support_network["ilim_resistor_ohm_max"]
        capacitors = list(_two_terminal(parts, pad_nets, "C"))
        in_cap = any({left[1], right[1]} == {vin_net, gnd[1]} for _, left, right in capacitors)
        out_cap = any({left[1], right[1]} == {out_net, gnd[1]} for _, left, right in capacitors)
        if not (chain_ok and ilim_ok and in_cap and out_cap):
            return False
    return True


def _source_profile(
    parts: list[dict[str, Any]],
    profile: str,
    connectors: list[dict[str, Any]],
    input_pairs: list[tuple[str, str]],
    graph: dict[str, set[str]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
    declared_sources: set[str] | None = None,
    minimum_contact_voltage_v: float | None = None,
) -> bool:
    """Prove source contact and return both land on the actual power input loop."""
    for connector in connectors:
        record = _record(connector)
        if record is None:
            continue
        features = record.physical_features
        if profile == "dc" and not features & {
            "power-connector", "wire-to-board-connector", "jst-connector",
            "pin-header", "header", "screw-terminal", "terminal-block", "barrel-jack-connector",
        }:
            continue
        if minimum_contact_voltage_v is not None:
            rating = record.operating_limits.get("voltage_v", record.operating_limits.get("voltage_max_v"))
            if (
                not isinstance(rating, (int, float)) or isinstance(rating, bool)
                or not math.isfinite(rating) or rating < minimum_contact_voltage_v
            ):
                continue
        if profile == "usb" and "usb-c-receptacle" not in features:
            continue
        if profile == "barrel" and "barrel-jack-connector" not in features:
            continue
        if profile == "terminal" and not ({"screw-terminal", "terminal-block"} & features):
            continue
        if profile == "header" and not ({"pin-header", "header"} & features):
            continue
        battery_features = {"power-connector", "wire-to-board-connector", "jst-connector", "jst-ph"}
        if declared_sources is not None:
            battery_features.update({"pin-header", "header"})
        if profile == "battery" and not (battery_features & features):
            continue
        pads = _all_pads(str(connector["ref"]), pad_nets)
        if profile == "usb":
            positive = {net for pin, net in pads if re.fullmatch(r"VBUS[0-9A-Z]*", names.get((str(connector["ref"]), pin), ""), re.I)}
            returns = {net for pin, net in pads if re.fullmatch(r"(GND|SHIELD)", names.get((str(connector["ref"]), pin), ""), re.I)}
            cc_nets = {net for pin, net in pads if re.fullmatch(r"CC[12]", names.get((str(connector["ref"]), pin), ""), re.I)}
            rd_nets = {
                cc_net
                for cc_net in cc_nets
                if any(
                    cc_net in {left[1], right[1]}
                    and (other := right[1] if left[1] == cc_net else left[1]) in returns
                    and (value := _resistance_ohms(str(resistor.get("value") or ""))) is not None
                    and 4_700 <= value <= 5_600
                    for resistor, left, right in _two_terminal(parts, pad_nets, "R")
                )
            }
            if len(cc_nets) != 2 or rd_nets != cc_nets:
                continue
        else:
            positive = {net for _, net in pads}
            returns = set(positive)
        if declared_sources is not None:
            positive &= declared_sources
        if any(
            source_net != return_net
            and _reaches(graph, source_net, input_net)
            and return_net == input_return
            for input_net, input_return in input_pairs
            for source_net in positive
            for return_net in returns
        ):
            return True
        # A passive fused distribution has no IC input pin.  Its source still
        # needs both contacts proved through real fuse branches to another
        # connector with the same physical return.
        if not input_pairs and any(
            source_net != return_net
            and return_net in {net for _, net in _all_pads(str(other["ref"]), pad_nets)}
            and any(_reaches(graph, source_net, net) and net != source_net for _, net in _all_pads(str(other["ref"]), pad_nets))
            for other in connectors if other is not connector
            for source_net in positive
            for return_net in returns
        ):
            return True
    return False


def _reverse_protection_observed(
    parts: list[dict[str, Any]], connectors: list[dict[str, Any]], input_pairs: list[tuple[str, str]],
    graph: dict[str, set[str]], names: dict[tuple[str, str], str], pad_nets: dict[tuple[str, str], str],
) -> bool:
    """Require source -> protective element -> delivered load, in that order."""
    for protector in parts:
        record = _record(protector)
        if record is None or "reverse-polarity-protection" not in record.physical_features:
            continue
        source = _one_port_pin(str(protector["ref"]), record, "input", names, pad_nets)
        protected = _one_port_pin(str(protector["ref"]), record, "output", names, pad_nets)
        ground = _converter_ground(protector, record, names, pad_nets)
        if not source or not protected or not ground or source[1] == protected[1]:
            continue
        source_connector = any(
            source[1] in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
            and ground in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
            for connector in connectors
        )
        load = any(
            _reaches(graph, protected[1], input_net) and input_net != source[1]
            for input_net, _ in input_pairs
        ) or any(
            protected[1] in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
            and ground in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
            and source[1] not in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
            for connector in connectors
        )
        if source_connector and load:
            return True
    for diode in (part for part in parts if _is_kind(part, "diode")):
        anode = _one_pin(str(diode["ref"]), names, pad_nets, r"^(A|ANODE)$")
        cathode = _one_pin(str(diode["ref"]), names, pad_nets, r"^(K|CATHODE)$")
        if not anode or not cathode or anode[1] == cathode[1]:
            continue
        source = any(anode[1] in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)} for connector in connectors)
        load = any(_reaches(graph, cathode[1], input_net) for input_net, _ in input_pairs)
        if source and load:
            return True
    return False


def _reverse_pmos_observed(
    parts: list[dict[str, Any]], connectors: list[dict[str, Any]], input_pairs: list[tuple[str, str]],
    names: dict[tuple[str, str], str], pad_nets: dict[tuple[str, str], str],
    *, device_ref: str | None = None,
) -> bool:
    """Prove the reviewed reverse-polarity PMOS direction and gate clamp."""
    for fet in _device(parts, "aonr21357"):
        if device_ref is not None and str(fet["ref"]) != device_ref:
            continue
        record = _record(fet)
        if record is None:
            continue
        ref = str(fet["ref"])
        source_bus = _all_reviewed_pins(ref, record, "source_pins", "source", names, pad_nets)
        drain_bus = _all_reviewed_pins(ref, record, "drain_pins", "drain", names, pad_nets)
        gate = _one_port_pin(ref, record, "gate", names, pad_nets)
        if not gate or len({net for _, net in source_bus}) != 1 or len({net for _, net in drain_bus}) != 1:
            continue
        protected_net, source_net = next(iter({net for _, net in source_bus})), next(iter({net for _, net in drain_bus}))
        if protected_net == source_net:
            continue
        source_returns = {
            return_net
            for connector in connectors
            if source_net in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
            for _, return_net in _all_pads(str(connector["ref"]), pad_nets)
            if return_net != source_net
        }
        grounded_gate = any(
            {left[1], right[1]} == {gate[1], return_net}
            and (value := _resistance_ohms(str(resistor.get("value") or ""))) is not None
            and 9_000 <= value <= 11_000
            for return_net in source_returns
            for resistor, left, right in _two_terminal(parts, pad_nets, "R")
        )
        clamped_gate = any(
            (_record(diode) and _record(diode).identity == "bzt52c10")
            and (anode := _one_pin(str(diode["ref"]), names, pad_nets, r"^(A|ANODE)$")) is not None
            and (cathode := _one_pin(str(diode["ref"]), names, pad_nets, r"^(K|CATHODE)$")) is not None
            and anode[1] == gate[1] and cathode[1] == protected_net
            for diode in parts
        )
        if grounded_gate and clamped_gate and any(input_net == protected_net for input_net, _ in input_pairs):
            return True
    return False


def _fused_outputs(
    parts: list[dict[str, Any]], connectors: list[dict[str, Any]], input_pairs: list[tuple[str, str]],
    pad_nets: dict[tuple[str, str], str],
) -> int:
    """Count distinct fuse-to-output-contact branches, excluding the source."""
    returns = {return_net for _, return_net in input_pairs}
    if not returns:
        connector_members: dict[str, set[str]] = defaultdict(set)
        for connector in connectors:
            for _, net in _all_pads(str(connector["ref"]), pad_nets):
                connector_members[net].add(str(connector["ref"]))
        returns = {net for net, refs in connector_members.items() if len(refs) >= 2}
    source_refs = {
        str(connector["ref"])
        for connector in connectors
        if any(
            source_net == input_net and return_net == input_return
            for _, source_net in _all_pads(str(connector["ref"]), pad_nets)
            for _, return_net in _all_pads(str(connector["ref"]), pad_nets)
            for input_net, input_return in input_pairs
            if source_net != return_net
        )
    }
    branches: set[str] = set()
    for fuse in parts:
        record = _record(fuse)
        if record is None or "fuse" not in record.physical_features:
            continue
        pads = _all_pads(str(fuse["ref"]), pad_nets)
        if len(pads) != 2 or pads[0][1] == pads[1][1]:
            continue
        if any(
            terminal_net == fuse_net
            and any(other_net in returns for other_pin, other_net in _all_pads(str(connector["ref"]), pad_nets) if other_pin != pin)
            for _, fuse_net in pads
            for connector in connectors if str(connector["ref"]) not in source_refs
            for pin, terminal_net in _all_pads(str(connector["ref"]), pad_nets)
        ):
            branches.add(str(fuse["ref"]))
    return len(branches)




def _reviewed_enable_pin(
    ref: str,
    record: Any,
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
) -> tuple[str, str] | None:
    """Resolve only a manufacturer-reviewed enable terminal on a delivered pad."""
    enable = (record.support_network or {}).get("enable")
    if not isinstance(enable, dict) or enable.get("float_enables") is not True:
        return None
    pin = str(enable.get("pin") or "")
    direct = (ref, pin)
    if pin and direct in pad_nets:
        return pin, pad_nets[direct]
    matches = _pins_for(ref, names, pad_nets, "^" + re.escape(pin) + "$")
    return matches[0] if pin and len(matches) == 1 else None


def _enable_jumper_observed(parts, connectors, names, pad_nets, *, device_ref: str | None = None) -> bool:
    ground_nets = {return_net for _, return_net in _power_input_pairs(parts, names, pad_nets)}
    ground_nets.update(
        net for (ref, pin), name in names.items()
        if re.fullmatch(r"(?:GND|VSS|AGND|DGND|PGND|0V)", name, re.I)
        and (net := pad_nets.get((ref, pin)))
    )
    jumpers = [part for part in parts if _is_reviewed_jumper(part)]
    for device in parts:
        if device_ref is not None and str(device["ref"]) != device_ref:
            continue
        record = _record(device)
        if record is None:
            continue
        enable = _reviewed_enable_pin(str(device["ref"]), record, names, pad_nets)
        ground = _converter_ground(device, record, names, pad_nets)
        if enable and ground and enable[1] != ground and any(
            len(pads := _all_pads(str(jumper["ref"]), pad_nets)) == 2
            and {pads[0][1], pads[1][1]} == {enable[1], ground}
            for jumper in jumpers
        ):
            return True
    # Distribution boards use the same physical jumper to pull a PMOS gate
    # through a resistor.  Verify the discrete control chain rather than a
    # writer-supplied role name.
    for jumper in jumpers:
        pads = _all_pads(str(jumper["ref"]), pad_nets)
        if len(pads) != 2:
            continue
        control = next((net for _, net in pads if net not in ground_nets), None)
        if control is None or not any(net in ground_nets for _, net in pads):
            continue
        ground = next(net for _, net in pads if net in ground_nets)
        for fet in _device(parts, "aonr21357"):
            if device_ref is not None and str(fet["ref"]) != device_ref:
                continue
            record = _record(fet)
            if record is None:
                continue
            ref = str(fet["ref"])
            source_bus = _all_reviewed_pins(ref, record, "source_pins", "source", names, pad_nets)
            drain_bus = _all_reviewed_pins(ref, record, "drain_pins", "drain", names, pad_nets)
            gate = _one_port_pin(ref, record, "gate", names, pad_nets)
            if not gate or len({net for _, net in source_bus}) != 1 or len({net for _, net in drain_bus}) != 1:
                continue
            source, drain = source_bus[0][1], drain_bus[0][1]
            if len({source, drain, gate[1], control, ground}) != 5:
                continue
            pullup = _resistor_path_value(source, gate[1], parts, pad_nets, maximum=6)
            pulldown = _resistor_path_value(gate[1], control, parts, pad_nets, maximum=1)
            if len(pullup) != 1 or len(pulldown) != 1:
                continue
            if not 9 * pulldown[0][0] <= pullup[0][0] <= 100_000:
                continue
            # Do not let this switch itself prove that its source is powered,
            # or reverse its source/drain orientation through an undirected edge.
            others = [part for part in parts if str(part["ref"]) != ref]
            graph = _power_edges(others, names, pad_nets, nonconverting_only=True)
            # This discrete voltage proof covers a standard 5 V USB input;
            # other input ranges remain unverified here, not guessed from labels.
            if not _source_profile(parts, "usb", connectors, [(source, ground)], graph, names, pad_nets):
                continue
            resistor_refs = set(pullup[0][1]) | set(pulldown[0][1])
            control_nets = {
                net for part in parts if str(part["ref"]) in resistor_refs
                for _, net in _all_pads(str(part["ref"]), pad_nets)
            } - {source, ground}
            control_ok = True
            for part in others:
                part_ref = str(part["ref"])
                if part_ref in resistor_refs or part_ref == str(jumper["ref"]):
                    continue
                if not control_nets.intersection(net for _, net in _all_pads(part_ref, pad_nets)):
                    continue
                clamp = _record(part)
                anode = _one_pin(part_ref, names, pad_nets, r"^(A|ANODE)$")
                cathode = _one_pin(part_ref, names, pad_nets, r"^(K|CATHODE)$")
                if not (
                    clamp and "zener-diode" in clamp.physical_features
                    and anode and cathode and anode[1] == gate[1] and cathode[1] == source
                    and 4.5 <= float(clamp.operating_limits.get("zener_voltage_min_v", 0))
                    <= float(clamp.operating_limits.get("zener_voltage_max_v", 0))
                    < float(record.operating_limits["vgs_abs_max_v"])
                ):
                    control_ok = False
                    break
            output_ok = any(
                ground in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
                and any(net != ground and _reaches(graph, drain, net)
                        for _, net in _all_pads(str(connector["ref"]), pad_nets))
                for connector in connectors if connector is not jumper
            )
            if control_ok and output_ok:
                return True
    return False


def _reset_button_observed(parts, names, pad_nets, powered_mcu_refs) -> bool:
    reset_nets = {
        net for (ref, pin), name in names.items()
        if (ref, pin) in pad_nets and re.fullmatch(r"(?:N?RESET|NRST|RST)", name, re.I)
        for net in [pad_nets[(ref, pin)]]
    }
    for part in parts:
        ref = str(part["ref"])
        record = _record(part)
        if ref not in powered_mcu_refs or record is None or "microcontroller" not in record.physical_features:
            continue
        # Reviewed MCU enable/run ports are reset controls even when their
        # native symbol calls the pin EN rather than RESET. A regulator EN
        # or an unpowered/incomplete MCU cannot supply this additional proof.
        for port in ("enable", "reset", "run"):
            if pin := _one_port_pin(ref, record, port, names, pad_nets):
                reset_nets.add(pin[1])
    ground_nets = {
        net for (ref, pin), name in names.items()
        if (ref, pin) in pad_nets and re.fullmatch(r"(?:GND|VSS|AGND|DGND|PGND|0V)", name, re.I)
        for net in [pad_nets[(ref, pin)]]
    }
    reset_nets -= ground_nets
    return any(
        len(pads := _all_pads(str(button["ref"]), pad_nets)) == 2
        and pads[0][1] != pads[1][1]
        and bool({pads[0][1], pads[1][1]} & reset_nets)
        and bool({pads[0][1], pads[1][1]} & ground_nets)
        for button in parts
        if {"button", "switch", "pushbutton", "momentary-button"} & _physical_features(button)
    )

def _declared_analog_reset_source_voltage(
    state: dict[str, Any], input_net: str, output_net: str | None = None,
) -> float | None:
    """Read a typed external-input contract bound to the delivered switch input.

    This is not a board voltage measurement.  It is the narrow case where the
    architecture explicitly contracts an off-board regulated 3.3 V or 5 V
    source and delivered copper proves that exact input net feeds the reset
    circuit.  Other labels, rails, and textual BOM claims remain unverified.
    """
    architecture = state.get("architecture") if isinstance(state, dict) else None
    requirements = architecture.get("requirements") if isinstance(architecture, dict) else None
    if not isinstance(requirements, list):
        return None
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        if str(requirement.get("family") or "").casefold().replace("_", "-") != "analog-power-cycle-reset":
            continue
        ports = requirement.get("ports")
        parameters = requirement.get("parameters")
        if (
            not isinstance(ports, dict)
            or not isinstance(parameters, dict)
            or ports.get("input") != input_net
            or output_net is not None and ports.get("output") != output_net
        ):
            continue
        voltage, current = parameters.get("rail_voltage_v"), parameters.get("source_current_a")
        if (
            isinstance(voltage, (int, float))
            and not isinstance(voltage, bool)
            and float(voltage) == 5.0
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
            and math.isclose(float(current), 0.5, rel_tol=0.0, abs_tol=1e-9)
        ):
            return float(voltage)
    return None


def _analog_reset_source_voltage(
    source_net: str,
    ground_net: str,
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
    state: dict[str, Any] | None = None,
    output_net: str | None = None,
) -> float | None:
    """Return a board-proven converter voltage or bound external-input contract."""
    for converter in parts:
        record = _record(converter)
        if record is None:
            continue
        input_pin = _one_port_pin(str(converter["ref"]), record, "input", names, pad_nets)
        feedback = _one_port_pin(str(converter["ref"]), record, "feedback", names, pad_nets)
        switch_pins = _record_port_pins(str(converter["ref"]), record, "switch", names, pad_nets)
        voltage = record.operating_limits.get("output_voltage_v")
        if (
            input_pin is None
            or ("feedback" in record.port_pins and feedback is None)
            or (feedback is not None and feedback[1] != source_net)
            or len(switch_pins) != 1
            or _converter_ground(converter, record, names, pad_nets) != ground_net
            or not isinstance(voltage, (int, float))
            or not 0 < float(voltage) <= 5.0
            or not _converter_support_ok(
                converter, record, input_pin[1], source_net, ground_net, parts, names, pad_nets
            )
        ):
            continue
        if any(
            {left[1], right[1]} == {switch_pins[0][1], source_net}
            for _, left, right in _two_terminal(parts, pad_nets, "L")
        ):
            return float(voltage)
    declared = _declared_analog_reset_source_voltage(state or {}, source_net, output_net)
    return declared if declared is not None and any(
        source_net in {net for _, net in _all_pads(str(connector["ref"]), pad_nets)}
        for connector in _connector(parts)
    ) else None


def _analog_power_cycle_reset_observed(parts, names, pad_nets, state) -> bool:
    """Prove supervised low-voltage analog rail power-cycling in delivered copper.

    The switch must have a guaranteed forward-off current bound, not merely a
    reverse-leakage rating. Independent supervision keeps signal isolation off
    until the switched rail recovers. Passive paths cannot bypass either cut.
    """
    for switch in _device(parts, "tps22948dckr"):
        record = _record(switch)
        if record is None:
            continue
        if record.operating_limits.get("off_input_current_max_a") != 5e-6:
            continue
        ref = str(switch["ref"])
        source = _one_port_pin(ref, record, "input", names, pad_nets)
        ground = _one_port_pin(ref, record, "ground", names, pad_nets)
        enable = _one_port_pin(ref, record, "enable", names, pad_nets)
        output = _one_port_pin(ref, record, "output", names, pad_nets)
        if not all((source, ground, enable, output)):
            continue
        source_net, ground_net, enable_net, rail = (
            source[1], ground[1], enable[1], output[1],
        )
        if len({source_net, ground_net, enable_net, rail}) != 4:
            continue
        feed_paths = _resistor_path_value(source_net, enable_net, parts, pad_nets, maximum=1)
        pull_paths = _resistor_path_value(enable_net, ground_net, parts, pad_nets, maximum=1)
        if not (
            len(feed_paths) == len(pull_paths) == 1
            and math.isclose(feed_paths[0][0], 5_900.0, rel_tol=0.01)
            and math.isclose(pull_paths[0][0], 60_000.0, rel_tol=0.01)
        ):
            continue
        input_caps = [
            _capacitance_farads(str(capacitor.get("value") or ""))
            for capacitor, left, right in _two_terminal(parts, pad_nets, "C")
            if {left[1], right[1]} == {source_net, ground_net}
        ]
        if not input_caps or any(value is None for value in input_caps) or sum(input_caps) < 1e-6:
            continue
        if not any(
            len(pads := _all_pads(str(button["ref"]), pad_nets)) == 2
            and {pads[0][1], pads[1][1]} == {enable_net, ground_net}
            for button in parts
            if "momentary-button" in _physical_features(button)
        ):
            continue
        source_voltage = _analog_reset_source_voltage(
            source_net, ground_net, parts, names, pad_nets, state, rail
        )
        if source_voltage != 5.0:
            continue
        if not any(
            math.isclose(
                _resistance_ohms(str(resistor.get("value") or "")) or 0.0,
                1_000.0,
                rel_tol=0.01,
            )
            and {left[1], right[1]} == {rail, ground_net}
            for resistor, left, right in _two_terminal(parts, pad_nets, "R")
        ):
            continue
        switched_capacitance = 0.0
        complete_capacitance = True
        for capacitor, left, right in _two_terminal(parts, pad_nets, "C"):
            terminals = {left[1], right[1]}
            if rail not in terminals:
                continue
            other = terminals - {rail}
            value = _capacitance_farads(str(capacitor.get("value") or ""))
            if len(other) != 1 or ground_net not in other or value is None:
                complete_capacitance = False
                break
            switched_capacitance += value
        if not complete_capacitance or switched_capacitance > 1e-6:
            continue
        expected_supervisor = "tps3808g50dbvr"
        supervisor_ok = False
        isolation_enable = None
        control_buffer_ref = None
        for supervisor in _device(parts, expected_supervisor):
            supervisor_record = _record(supervisor)
            if supervisor_record is None:
                continue
            supervisor_ref = str(supervisor["ref"])
            reset = _one_port_pin(supervisor_ref, supervisor_record, "reset", names, pad_nets)
            supervisor_ground = _one_port_pin(supervisor_ref, supervisor_record, "ground", names, pad_nets)
            manual_reset = _one_port_pin(supervisor_ref, supervisor_record, "manual_reset", names, pad_nets)
            ct = _one_port_pin(supervisor_ref, supervisor_record, "ct", names, pad_nets)
            sense = _one_port_pin(supervisor_ref, supervisor_record, "sense", names, pad_nets)
            vdd = _one_port_pin(supervisor_ref, supervisor_record, "vdd", names, pad_nets)
            if not all((reset, supervisor_ground, manual_reset, sense, vdd)):
                continue
            reset_pullups = _resistor_path_value(reset[1], enable_net, parts, pad_nets, maximum=1)
            supervisor_caps = [
                _capacitance_farads(str(capacitor.get("value") or ""))
                for capacitor, left, right in _two_terminal(parts, pad_nets, "C")
                if {left[1], right[1]} == {source_net, ground_net}
            ]
            if (
                reset[1] == enable_net
                or supervisor_ground[1] != ground_net
                or manual_reset[1] != source_net
                or sense[1] != rail
                or vdd[1] != source_net
                or ct is not None
                or len(reset_pullups) != 1
                or not 10_000.0 <= reset_pullups[0][0] <= 100_000.0
                or not supervisor_caps
                or any(value is None for value in supervisor_caps)
                or sum(value for value in supervisor_caps if value is not None) < 100e-9
            ):
                continue
            # The supervisor's resistive pull-up is too slow for 1G66's
            # 10 ns/V control-input limit. Require the actual Schmitt buffer.
            for buffer in _device(parts, "sn74lvc1g17dbvr"):
                buffer_ref = str(buffer["ref"])
                buffer_record = _record(buffer)
                buffer_input = _one_port_pin(buffer_ref, buffer_record, "input", names, pad_nets)
                buffer_output = _one_port_pin(buffer_ref, buffer_record, "output", names, pad_nets)
                buffer_vdd = _one_port_pin(buffer_ref, buffer_record, "vdd", names, pad_nets)
                buffer_ground = _one_port_pin(buffer_ref, buffer_record, "ground", names, pad_nets)
                buffer_nc = _one_port_pin(buffer_ref, buffer_record, "no_connect", names, pad_nets)
                if (
                    buffer_input and buffer_input[1] == reset[1]
                    and buffer_output
                    and buffer_output[1] not in {reset[1], enable_net, source_net, ground_net, rail}
                    and buffer_vdd and buffer_vdd[1] == source_net
                    and buffer_ground and buffer_ground[1] == ground_net
                    and buffer_nc is None
                    and sum(value >= 100e-9 for value in supervisor_caps if value is not None) >= 3
                ):
                    isolation_enable = buffer_output[1]
                    control_buffer_ref = buffer_ref
                    break
            if isolation_enable is None:
                continue
            supervisor_ok = True
            break
        if not supervisor_ok or isolation_enable is None:
            continue
        analog_loads = [
            part for part in parts
            if part is not switch
            and any(
                feature == "op-amp" or "operational-amplifier" in feature or "sensor" in feature
                for feature in _physical_features(part)
            )
        ]
        if not analog_loads:
            continue
        analog_drive_nets = {
            pad_nets[(str(load["ref"]), str(pin))]
            for load in analog_loads
            if (load_record := _record(load)) is not None
            for port, pin in load_record.port_pins.items()
            if (port == "output" or port.startswith("output_"))
            and (str(load["ref"]), str(pin)) in pad_nets
        }
        load_source_nets = analog_drive_nets | {rail}
        passive_load_a = 0.0
        passive_loads_bounded = True
        for resistor, left, right in _two_terminal(parts, pad_nets, "R"):
            terminals = {left[1], right[1]}
            if not terminals & load_source_nets:
                continue
            resistance = _resistance_ohms(str(resistor.get("value") or ""))
            if resistance is None or resistance <= 0:
                passive_loads_bounded = False
                break
            # Quiescent device current excludes output drive. Bound every
            # rail- or analog-output-touched resistor at the full rail voltage,
            # including every discharge resistor rather than only one bleed.
            passive_load_a += source_voltage / resistance
        # A trimmer used as a direct rail/output load can be adjusted to zero
        # ohms. The reviewed lowerer puts one only behind a fixed series
        # feedback/divider resistor; without that physical separation the
        # 100 mA budget has no finite lower-bound resistance and must refuse.
        for trimmer in (
            part for part in parts
            if (trimmer_record := _record(part)) is not None
            and trimmer_record.identity == "3296w-1-103lf"
        ):
            trimmer_ref = str(trimmer["ref"])
            ccw = pad_nets.get((trimmer_ref, "1"))
            wiper = pad_nets.get((trimmer_ref, "2"))
            cw = pad_nets.get((trimmer_ref, "3"))
            if not ccw or not wiper or wiper != cw or ccw == wiper:
                passive_loads_bounded = False
                break
            if {ccw, wiper} & load_source_nets:
                passive_loads_bounded = False
                break
        if not passive_loads_bounded:
            continue
        # Follow passive signal paths, but never walk through ground. Include
        # other reviewed power paths to reject a parallel feed around the switch.
        conductive = _passive_edges(parts, pad_nets)
        for net, peers in _power_edges(
            [part for part in parts if str(part["ref"]) != ref], names, pad_nets
        ).items():
            conductive[net].update(peers)
        conductive.pop(ground_net, None)
        for peers in conductive.values():
            peers.discard(ground_net)
        if _reaches(conductive, source_net, rail):
            continue
        analog_signal_nets = {
            net
            for load in analog_loads
            for _, net in _all_pads(str(load["ref"]), pad_nets)
            if net not in {ground_net, rail}
        }
        pending = list(analog_signal_nets)
        while pending:
            net = pending.pop()
            for peer in conductive.get(net, ()):
                if peer != rail and peer not in analog_signal_nets:
                    analog_signal_nets.add(peer)
                    pending.append(peer)
        isolators = _device(parts, "sn74lvc1g66dbvr")
        if len(isolators) > 16:
            continue
        trim_refs = {
            str(part["ref"])
            for part in parts
            if (record := _record(part)) is not None
            and record.identity == "3296w-1-103lf"
        }
        protected_refs = {str(part["ref"]) for part in (*analog_loads, *isolators)} | trim_refs
        foreign_nets = {
            net
            for part in parts
            if str(part["ref"]) not in protected_refs
            and _symbol_kind(part) not in {"resistor", "capacitor", "inductor"}
            for _, net in _all_pads(str(part["ref"]), pad_nets)
            if net not in {ground_net, rail}
        }
        if analog_signal_nets & foreign_nets:
            continue
        isolated = True
        for isolator in isolators:
            isolator_ref = str(isolator["ref"])
            isolator_record = _record(isolator)
            signal_a = _one_port_pin(isolator_ref, isolator_record, "signal_a", names, pad_nets)
            signal_b = _one_port_pin(isolator_ref, isolator_record, "signal_b", names, pad_nets)
            if not any(signal and signal[1] in analog_signal_nets for signal in (signal_a, signal_b)):
                continue
            control = _one_port_pin(isolator_ref, isolator_record, "control", names, pad_nets)
            vdd = _one_port_pin(isolator_ref, isolator_record, "vdd", names, pad_nets)
            return_pin = _one_port_pin(isolator_ref, isolator_record, "ground", names, pad_nets)
            if not (
                signal_a and signal_b and signal_a[1] != signal_b[1]
                and control and control[1] == isolation_enable
                and vdd and vdd[1] == source_net
                and return_pin and return_pin[1] == ground_net
                and isolator_record is not None
                and isolator_record.operating_limits.get("off_leakage_max_a") == 1e-6
            ):
                isolated = False
                break
        if not isolated:
            continue
        # This private output drives only the reviewed isolation-control pins;
        # an external driver or capacitive load invalidates the edge contract.
        control_contacts = {(control_buffer_ref, "4")} | {
            (str(isolator["ref"]), "4")
            for isolator in isolators
            if pad_nets.get((str(isolator["ref"]), "4")) == isolation_enable
        }
        if {
            contact for contact, net in pad_nets.items() if net == isolation_enable
        } != control_contacts:
            continue
        load_current_a = 0.0
        powered = True
        for load in analog_loads:
            load_ref = str(load["ref"])
            load_record = _record(load)
            positive = [
                (str(pin), pad_nets[(load_ref, str(pin))])
                for port, pin in (load_record.port_pins.items() if load_record is not None else ())
                if port in {"vdd", "vcc", "vplus", "vsplus"} and (load_ref, str(pin)) in pad_nets
            ]
            negative = [
                (str(pin), pad_nets[(load_ref, str(pin))])
                for port, pin in (load_record.port_pins.items() if load_record is not None else ())
                if port in {"ground", "gnd", "vss", "vminus", "vsminus"} and (load_ref, str(pin)) in pad_nets
            ]
            current = load_record.operating_limits.get("supply_current_max_a") if load_record is not None else None
            supply_minimum = load_record.operating_limits.get("supply_min_v") if load_record is not None else None
            supply_maximum = load_record.operating_limits.get("supply_max_v") if load_record is not None else None
            if (
                not positive
                or not negative
                or any(net != rail for _, net in positive)
                or any(net != ground_net for _, net in negative)
                or not isinstance(supply_minimum, (int, float))
                or not isinstance(supply_maximum, (int, float))
                or not supply_minimum <= source_voltage <= supply_maximum
                or not isinstance(current, (int, float))
                or isinstance(current, bool)
                or current < 0
            ):
                powered = False
                break
            load_current_a += float(current)
        if powered and load_current_a + passive_load_a <= 0.100:
            return True
    return False


def _declared_dual_source_range(state, positive, ground, source_class, minimum, maximum):
    """Bind a typed external source range to the already-proven connector nets."""
    architecture = state.get("architecture") or {}
    for requirement in architecture.get("requirements", []):
        if requirement.get("family") != "dual-source-battery-power":
            continue
        ports, parameters = requirement.get("ports") or {}, requirement.get("parameters") or {}
        prefix = "host" if source_class == "host_5v" else "battery"
        if ports.get(f"{prefix}_input") != positive or ports.get("gnd") != ground:
            continue
        if prefix == "host" and parameters.get("host_voltage_v") != 5.0:
            continue
        if prefix == "battery" and parameters.get("battery_chemistry") != "single-cell-liion":
            continue
        low, high = parameters.get(f"{prefix}_voltage_min_v"), parameters.get(f"{prefix}_voltage_max_v")
        if (
            isinstance(low, (int, float)) and not isinstance(low, bool)
            and isinstance(high, (int, float)) and not isinstance(high, bool)
            and minimum <= low <= high <= maximum
        ):
            return True
    return False


def _supply_voltage_observed(net, ground, parts, names, pad_nets, state, outputs):
    """Prove conversion or bind a declared external source to physical contacts.

    An external-source voltage is an interface contract, not a measurement.
    Net spelling and an unbound architecture rail have no authority.
    """
    voltages = {
        row["voltage"] for row in outputs
        if row["output_net"] == net and row["ground_net"] == ground
    }
    connectors = _connector(parts)
    graph = _power_edges(parts, names, pad_nets, nonconverting_only=True)
    if _source_profile(parts, "usb", connectors, [(net, ground)], graph, names, pad_nets):
        voltages.add(5.0)
    architecture = state.get("architecture") or {}
    rails = architecture.get("rail_voltages") or {}
    for requirement in architecture.get("requirements", []):
        contacts = set((requirement.get("ports") or {}).values())
        if requirement.get("role") != "power_input" or ground not in contacts:
            continue
        for source in contacts - {ground}:
            declared = rails.get(source)
            if (
                not isinstance(declared, (int, float)) or isinstance(declared, bool)
                or not math.isfinite(declared) or declared <= 0
                or not _reaches(graph, source, net)
            ):
                continue
            if any(
                {source, ground} <= {value for _, value in _all_pads(str(connector["ref"]), pad_nets)}
                and str(connector.get("recipe_role") or "") != "program_header"
                and str(connector.get("value") or "").strip().upper() != "UART PROGRAM"
                for connector in connectors
            ):
                voltages.add(float(declared))
    return next(iter(voltages)) if len(voltages) == 1 else None


def _observed_linear_outputs(parts, names, pad_nets, state, outputs):
    """Observe enabled nominal linear outputs, not load-current/thermal ratings.

    Resolve upstream sources before downstream supplies. Cyclic, unpowered or
    under-headroom chains cannot bootstrap themselves into voltage evidence.
    """
    observed = []
    pending = [
        (part, record) for part in parts
        if (record := _record(part)) is not None
        and "regulated_input_min_v" in record.operating_limits
        and record.physical_features & {"ldo-regulator", "linear-regulator"}
    ]
    capacitors = [
        ({left[1], right[1]}, _capacitance_farads(str(part.get("value") or "")))
        for part, left, right in _two_terminal(parts, pad_nets, "C")
    ]
    while pending:
        remaining = []
        for part, record in pending:
            ref = str(part["ref"])
            source = _one_port_pin(ref, record, "input", names, pad_nets)
            output = _one_port_pin(ref, record, "output", names, pad_nets)
            ground = _one_port_pin(ref, record, "ground", names, pad_nets)
            enable = _one_port_pin(ref, record, "enable", names, pad_nets)
            if (
                not all((source, output, ground, enable))
                or len({source[1], output[1], ground[1]}) != 3
                or enable[1] != source[1]
                or _one_port_pin(ref, record, "no_connect", names, pad_nets) is not None
            ):
                continue
            limits = record.operating_limits
            if not all(
                any(nets == {rail, ground[1]} and value is not None and value >= minimum
                    for nets, value in capacitors)
                for rail, minimum in (
                    (source[1], limits["input_bypass_min_f"]),
                    (output[1], limits["output_bypass_min_f"]),
                )
            ):
                continue
            minimum_load = float(limits.get("regulated_output_min_load_a") or 0)
            if minimum_load:
                # Bound the delivered preload at -5% output and +20% resistance;
                # an unspecified off-board load cannot qualify a regulator.
                preload = sum(
                    float(limits["output_voltage_v"]) * 0.95 / (resistance * 1.20)
                    for resistor, left, right in _two_terminal(parts, pad_nets, "R")
                    if {left[1], right[1]} == {output[1], ground[1]}
                    and (resistance := _resistance_ohms(str(resistor.get("value") or ""))) is not None
                    and resistance > 0
                )
                if preload < minimum_load:
                    continue
            voltage = _supply_voltage_observed(
                source[1], ground[1], parts, names, pad_nets, state, [*outputs, *observed],
            )
            if voltage is None:
                remaining.append((part, record))
                continue
            if limits["regulated_input_min_v"] <= voltage <= limits["input_voltage_max_v"]:
                observed.append({
                    "ref": ref, "input_net": source[1], "output_net": output[1],
                    "ground_net": ground[1], "voltage": float(limits["output_voltage_v"]),
                    "current_a": None,
                })
        if len(remaining) == len(pending):
            break
        pending = remaining
    return observed


def _addressable_rgb_status_refs(parts, names, pad_nets, state, outputs, complete_mcu_refs):
    """Prove powered V5 pixels driven through an enabled, bypassed TTL buffer."""
    gpio_pins = reviewed_application_pins("output")
    pixels = _device(parts, "ws2812b-b/t")
    if not pixels:
        return set()
    resistors = list(_two_terminal(parts, pad_nets, "R"))
    capacitors = list(_two_terminal(parts, pad_nets, "C"))
    proven: set[str] = set()
    for buffer in _device(parts, "sn74ahct1g125dbvr"):
        ref = str(buffer["ref"])
        pins = dict(_all_pads(ref, pad_nets))
        if not all(pins.get(pin) for pin in ("1", "2", "3", "4", "5")):
            continue
        enable, control, ground, shifted, rail = (pins[str(pin)] for pin in range(1, 6))
        if enable != ground or len({control, ground, shifted, rail}) != 4:
            continue
        voltage = _supply_voltage_observed(rail, ground, parts, names, pad_nets, state, outputs)
        if voltage is None or not 4.5 <= voltage <= 5.3:
            continue
        sources = [
            (part, record, pin)
            for part in parts if (record := _record(part)) is not None
            for pin in gpio_pins.get(record.identity, ())
            if pad_nets.get((str(part["ref"]), pin)) == control
        ]
        if len(sources) != 1:
            continue
        source, record, _ = sources[0]
        source_ref = str(source["ref"])
        if source_ref not in complete_mcu_refs:
            continue
        supplies = _record_port_pins(source_ref, record, "vdd", names, pad_nets) or _pins_for(
            source_ref, names, pad_nets, r"^(IOVDD|VDDIO|VDD|VDD_3V3|VCC)$"
        )
        returns = (
            _record_port_pins(source_ref, record, "ground", names, pad_nets)
            or _record_port_pins(source_ref, record, "vss", names, pad_nets)
            or _pins_for(source_ref, names, pad_nets, r"^(GND|VSS)$")
        )
        if not supplies or {net for _, net in returns} != {ground}:
            continue
        io_voltages = {
            _supply_voltage_observed(net, ground, parts, names, pad_nets, state, outputs)
            for _, net in supplies
        }
        if (
            len(io_voltages) != 1 or None in io_voltages
            or not math.isclose(next(iter(io_voltages)), 3.3, abs_tol=0.05)
            or record.operating_limits.get("output_high_min_v_at_3v3", 0) < 2.0
            or record.operating_limits.get("output_low_max_v_at_3v3", math.inf) > 0.8
        ):
            continue
        if not any(
            {left[1], right[1]} == {control, ground}
            and (value := _resistance_ohms(str(resistor.get("value") or ""))) is not None
            and 10_000 <= value <= 100_000
            for resistor, left, right in resistors
        ):
            continue
        starts = {
            right[1] if left[1] == shifted else left[1]
            for resistor, left, right in resistors
            if shifted in {left[1], right[1]}
            and (value := _resistance_ohms(str(resistor.get("value") or ""))) is not None
            and 100 <= value <= 470
        }
        if len(starts) != 1:
            continue
        data = next(iter(starts))
        visited: set[str] = set()
        malformed = False
        while data:
            if data in {rail, ground, control, shifted}:
                malformed = True
                break
            attached = [
                pixel for pixel in pixels
                if pad_nets.get((str(pixel["ref"]), "4")) == data
            ]
            if not attached:
                break
            if len(attached) != 1:
                malformed = True
                break
            pixel_ref = str(attached[0]["ref"])
            if (
                pixel_ref in visited
                or pad_nets.get((pixel_ref, "1")) != rail
                or pad_nets.get((pixel_ref, "3")) != ground
                or data in {rail, ground, control, shifted}
            ):
                malformed = True
                break
            visited.add(pixel_ref)
            data = pad_nets.get((pixel_ref, "2"))
        bypasses = sum(
            {left[1], right[1]} == {rail, ground}
            and (value := _capacitance_farads(str(capacitor.get("value") or ""))) is not None
            and 100e-9 <= value <= 220e-9
            for capacitor, left, right in capacitors
        )
        if visited and not malformed and bypasses >= len(visited) + 1:
            proven.update(visited)
    return proven


def _rgb_power_context(parts, connectors, names, pad_nets, state, outputs, graph):
    """Bind passive distribution sense nodes, without asserting output voltage."""
    inputs = set(_power_input_pairs(parts, names, pad_nets))
    context_outputs = list(outputs)
    for requirement in (state.get("architecture") or {}).get("requirements", []):
        if requirement.get("family") != "fused-power-distribution":
            continue
        ports = requirement.get("ports") or {}
        positive, ground = ports.get("input"), ports.get("gnd")
        if not (
            isinstance(positive, str) and isinstance(ground, str)
            and positive != ground
            and _supply_voltage_observed(
                positive, ground, parts, names, pad_nets, state, outputs
            ) is not None
        ):
            continue
        branch_nets = {
            net for port, net in ports.items()
            if re.fullmatch(r"output[1-9][0-9]*", port) and isinstance(net, str)
        }
        if not branch_nets or branch_nets & {positive, ground}:
            continue
        if not all(
            _reaches(graph, positive, net)
            and any(
                {net, ground} <= {value for _, value in _all_pads(str(connector["ref"]), pad_nets)}
                for connector in connectors
            )
            for net in branch_nets
        ):
            continue
        inputs.add((positive, ground))
        context_outputs.extend(
            {"input_net": positive, "output_net": net, "ground_net": ground}
            for net in sorted(branch_nets)
        )
    return inputs, context_outputs


def _rgb_status_observed(parts, input_pairs, outputs, names, pad_nets) -> bool:
    power_nets = {net for pair in input_pairs for net in pair} | {
        net for output in outputs for net in (output["input_net"], output["output_net"], output["ground_net"])
    }
    grounds = {pair[1] for pair in input_pairs} | {
        output["ground_net"] for output in outputs
    }
    source_nets = power_nets - grounds
    # A reviewed passive fuse preserves the input domain. Do not promote
    # arbitrary power-transfer devices or back-propagate claimed output rails.
    fuse_graph = _power_edges(
        [part for part in parts if "fuse" in _physical_features(part)],
        names, pad_nets, nonconverting_only=True,
    )
    protected_inputs = {positive for positive, _ in input_pairs} - grounds
    pending = list(protected_inputs)
    for positive in pending:
        for peer in fuse_graph.get(positive, ()):
            if peer not in grounds and peer not in protected_inputs:
                protected_inputs.add(peer)
                pending.append(peer)
    source_nets.update(protected_inputs)
    power_nets.update(protected_inputs)

    # A base chain may contain as many resistors as the voltage-rated design
    # requires. Walk only simple, physically isolated series interiors; never
    # use ground or another rail as a stepping stone to a control source.
    resistor_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for resistor, left, right in _two_terminal(parts, pad_nets, "R"):
        value = _resistance_ohms(str(resistor.get("value") or ""))
        if value is not None and value > 0:
            ref = str(resistor["ref"])
            resistor_edges[left[1]].append((right[1], ref))
            resistor_edges[right[1]].append((left[1], ref))
    pad_counts: dict[str, int] = defaultdict(int)
    for net in pad_nets.values():
        if net:
            pad_counts[net] += 1

    def source_chain(base: str, targets: set[str] = source_nets) -> int:
        if base in power_nets:
            return 0
        found: list[int] = []
        for first, ref in resistor_edges.get(base, ()):
            net = first
            seen = {base}
            while net not in seen:
                seen.add(net)
                if net in power_nets:
                    if net in targets:
                        found.append(len(seen) - 1)
                    break
                edges = resistor_edges.get(net, ())
                if len(edges) != 2 or pad_counts[net] != 2:
                    break
                onward = [(other, edge_ref) for other, edge_ref in edges if edge_ref != ref]
                if len(onward) != 1:
                    break
                net, ref = onward[0]
        return found[0] if len(found) == 1 else 0

    def controlled_base(base: str) -> bool:
        if not source_chain(base):
            return False
        # A transistor pulling the base down is an inverter, not an
        # independent source. Its own output-sense chain and return must exist.
        inverters = []
        for part in parts:
            ref = str(part["ref"])
            collector = _one_pin(ref, names, pad_nets, r"^(C|COLLECTOR)$")
            if collector and collector[1] == base:
                inverters.append(part)
        for inverter in inverters:
            ref = str(inverter["ref"])
            record = _record(inverter)
            emitter = _one_pin(ref, names, pad_nets, r"^(E|EMITTER)$")
            sense = _one_pin(ref, names, pad_nets, r"^(B|BASE)$")
            if (
                record is None or "npn-transistor" not in record.physical_features
                or not emitter or emitter[1] not in grounds or not sense
                or not source_chain(sense[1], {output["output_net"] for output in outputs})
                or not any(
                    other in grounds for other, _ in resistor_edges.get(sense[1], ())
                )
            ):
                return False
        return len(inverters) <= 1

    def meaningful_terminal(net: str) -> bool:
        if net in power_nets:
            return True
        for part in parts:
            collector = _one_pin(str(part["ref"]), names, pad_nets, r"^(C|COLLECTOR)$")
            emitter = _one_pin(str(part["ref"]), names, pad_nets, r"^(E|EMITTER)$")
            base = _one_pin(str(part["ref"]), names, pad_nets, r"^(B|BASE)$")
            record = _record(part)
            if (
                collector and emitter and base and collector[1] == net
                and emitter[1] in grounds
                and controlled_base(base[1])
                and (
                    source_chain(base[1]) <= 3
                    or (record is not None and "npn-transistor" in record.physical_features)
                )
            ):
                return True
            fault = _one_port_pin(str(part["ref"]), record, "fault", names, pad_nets) if record else None
            if fault and fault[1] == net:
                return True
        return False

    for led in _device(parts, "aptf1616lseezgkqbkc", "xl-1615rgbc-rf"):
        record = _record(led)
        common = _one_port_pin(str(led["ref"]), record, "common_anode", names, pad_nets) if record else None
        cathodes = [
            _one_port_pin(str(led["ref"]), record, port, names, pad_nets) if record else None
            for port in ("red_cathode", "green_cathode", "blue_cathode")
        ]
        if not common or common[1] not in source_nets or any(cathode is None for cathode in cathodes):
            continue
        if len({cathode[1] for cathode in cathodes}) != 3 or any(
            cathode[1] == common[1] for cathode in cathodes
        ):
            continue
        ends: set[str] = set()
        for cathode in cathodes:
            terminals = {
                net
                for net in {net for net in pad_nets.values() if net and net != common[1]}
                if meaningful_terminal(net)
                and _resistor_path_value(cathode[1], net, parts, pad_nets, maximum=3)
            }
            if len(terminals) != 1:
                break
            ends.update(terminals)
        else:
            if len(ends) < 2:
                continue
            drivers = []
            fault_terminal = False
            for part in parts:
                ref = str(part["ref"])
                record = _record(part)
                fault = _one_port_pin(ref, record, "fault", names, pad_nets) if record else None
                fault_terminal |= bool(fault and fault[1] in ends)
                collector = _one_pin(ref, names, pad_nets, r"^(C|COLLECTOR)$")
                base = _one_pin(ref, names, pad_nets, r"^(B|BASE)$")
                if collector and base and collector[1] in ends:
                    drivers.append(base[1])
            # The newly supported long-chain discrete status topology needs
            # its output-sense inverter; otherwise two identical input-driven
            # channels could masquerade as input/output/fault indication.
            if not fault_terminal and any(source_chain(base) > 3 for base in drivers):
                sense_pairs = []
                for part in parts:
                    ref = str(part["ref"])
                    collector = _one_pin(ref, names, pad_nets, r"^(C|COLLECTOR)$")
                    sense = _one_pin(ref, names, pad_nets, r"^(B|BASE)$")
                    if collector and sense and collector[1] in drivers:
                        sense_pairs.append((collector[1], sense[1]))
                if not any(
                    source_chain(driver, {output["output_net"]})
                    and source_chain(sense, {output["output_net"]})
                    and source_chain(inverted, {common[1]})
                    and any(
                        other not in {inverted, driver} and source_chain(other, {common[1]})
                        for other in drivers
                    )
                    for inverted, sense in sense_pairs
                    for driver in drivers if driver != inverted
                    for output in outputs
                ):
                    continue
            return True
    return False
def _power_control_gates(
    parts: list[dict[str, Any]], connectors: list[dict[str, Any]], names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str], requested: set[str], state: dict[str, Any],
    outputs: list[dict[str, Any]], internal_outputs: list[dict[str, Any]],
) -> dict[str, str]:
    """Emit only contract-requested power gates that physical pads prove."""
    graph = _power_edges(parts, names, pad_nets, nonconverting_only=True)
    for output in internal_outputs:
        # Conversions are directed; a linear regulator is not a voltage-preserving wire.
        graph[output["input_net"]].add(output["output_net"])
    inputs = _power_input_pairs(parts, names, pad_nets)
    gates: dict[str, str] = {}
    profiles = {
        "external_usb_c_5v_input_binding": "usb", "external_12v_barrel_jack_input_binding": "barrel",
        "external_18v_dc_input_binding": "dc", "external_24v_screw_terminal_input_binding": "terminal",
        "external_host_5v_header_input_binding": "header",
    }
    return_safe = _tps26600_return_isolated(parts, names, pad_nets) and _tps26600_support_observed(parts, names, pad_nets)
    architecture = state.get("architecture") or {}
    voltages = architecture.get("rail_voltages") or {}
    host_5v_sources = {
        net for requirement in architecture.get("requirements", [])
        if requirement.get("role") == "power_input"
        for net in (requirement.get("ports") or {}).values()
        if isinstance(net, str) and voltages.get(net) == 5.0
    }
    dc_18v_sources = {
        net for requirement in architecture.get("requirements", [])
        if requirement.get("role") == "power_input"
        for net in (requirement.get("ports") or {}).values()
        if isinstance(net, str) and voltages.get(net) == 18.0
    }
    for gate, profile in profiles.items():
        if gate not in requested or not return_safe:
            continue
        declared = (
            host_5v_sources if gate == "external_host_5v_header_input_binding"
            else dc_18v_sources if gate == "external_18v_dc_input_binding" else None
        )
        if declared is not None and not declared:
            continue
        if _source_profile(
            parts, profile, connectors, inputs, graph, names, pad_nets,
            declared_sources=declared,
            minimum_contact_voltage_v=18.0 if gate == "external_18v_dc_input_binding" else None,
        ):
            gates[gate] = "pass"
    # A generic header can be the board-side battery contact. Chemistry is an
    # external-source contract, not something connector shape or a net name proves.
    battery_sources = {
        net for requirement in architecture.get("requirements", [])
        if requirement.get("role") == "power_input"
        for net in (requirement.get("ports") or {}).values()
        if isinstance(net, str) and voltages.get(net) in (7.2, 7.4)
    }
    if (
        "external_2s_liion_pack_source_binding" in requested and return_safe
        and battery_sources
        and _source_profile(
            parts, "battery", connectors, inputs, graph, names, pad_nets,
            declared_sources=battery_sources,
        )
    ):
        gates["external_2s_liion_pack_source_binding"] = "pass"
    if "external_reverse_polarity_protection" in requested and return_safe and (
        _reverse_protection_observed(parts, connectors, inputs, graph, names, pad_nets)
        or _reverse_pmos_observed(parts, connectors, inputs, names, pad_nets)
    ):
        gates["external_reverse_polarity_protection"] = "pass"
    for voltage, gate in ((3.3, "external_conversion_to_3.3v_binding"), (5.0, "external_conversion_to_5v_binding"), (12.0, "external_conversion_to_12v_binding")):
        if gate in requested and any(output.get("converting") and abs(output["voltage"] - voltage) <= 0.05 and _source_profile(
            parts, profile, connectors, [(output["input_net"], output["ground_net"])], graph, names, pad_nets
        ) for profile in ("usb", "barrel", "terminal", "header", "battery") for output in outputs):
            gates[gate] = "pass"
    fused_count = _fused_outputs(parts, connectors, inputs, pad_nets)
    for required, gate in ((2, "external_fused_output_count_2"), (4, "external_fused_output_count_4"), (8, "external_fused_output_count_8")):
        if gate in requested and fused_count >= required:
            gates[gate] = "pass"
    if "external_power_distribution_function" in requested and fused_count >= 2:
        gates["external_power_distribution_function"] = "pass"
    return gates
def _select_output_voltage(observed: list[tuple[str, float]]) -> float | None:
    """Publish a scalar only when exactly one physical output is exposed."""
    return observed[0][1] if len(observed) == 1 else None


def _reviewed_led_current_loop(
    parts: list[dict[str, Any]],
    names: dict[tuple[str, str], str],
    pad_nets: dict[tuple[str, str], str],
    *,
    controller_ref: str | None = None,
) -> float | None:
    """Return a board-proven current only for the reviewed complete LED loop.

    The component list was already reconciled against actual footprint IDs.  This
    adds the remaining physical proof: every topology endpoint below is an
    observed pad/net relation, never a BOM net label or a textual circuit claim.
    """
    for controller in parts:
        if controller_ref is not None and str(controller.get("ref")) != controller_ref:
            continue
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
                cathode = _one_pin(str(diode["ref"]), names, pad_nets, r"^(K|C|CATHODE)$")
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
                and (
                    "high-power-led" not in _physical_features(led)
                    or pad_nets.get((str(led["ref"]), "3")) == cathode[1]
                )
                for led in parts
                if _is_kind(led, "led") or "high-power-led" in _physical_features(led)
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


def _ch340c_support_observed(bridge, parts, pad_nets, vbus, vdd, ground, observed_outputs) -> bool:
    """Observe the manufacturer's separate 3.3-V and 5-V V3 connections."""
    ref = str(bridge["ref"])
    v3 = pad_nets.get((ref, "4"))
    if pad_nets.get((ref, "1")) != ground or not v3:
        return False
    capacitors = [
        ({left[1], right[1]}, _capacitance_farads(str(capacitor.get("value") or "")))
        for capacitor, left, right in _two_terminal(parts, pad_nets, "C")
    ]
    if not any(
        nets == {vdd, ground} and value is not None and value >= 100e-9
        for nets, value in capacitors
    ):
        return False
    voltages = {
        output["voltage"] for output in observed_outputs
        if output["output_net"] == vdd and output["ground_net"] == ground
    }
    if vdd == vbus:
        voltages.add(5.0)
    if len(voltages) != 1:
        return False
    voltage = next(iter(voltages))
    if 3.1 <= voltage <= 3.6:
        return v3 == vdd
    return (
        4.0 <= voltage <= 5.3 and v3 not in {vdd, ground}
        and any(
            nets == {v3, ground} and value is not None and value >= 100e-9
            for nets, value in capacitors
        )
    )


def _cp2102n_support_observed(bridge, parts, pad_nets, vbus, vdd, ground, observed_outputs) -> bool:
    """Prove the reviewed external-3.3-V supply and protected VBUS sense circuit."""
    ref = str(bridge["ref"])
    sense = pad_nets.get((ref, "8"))
    reset = pad_nets.get((ref, "9"))
    if not sense or not reset or len({vbus, vdd, ground, sense, reset}) != 5:
        return False
    if pad_nets.get((ref, "7")) != vdd or pad_nets.get((ref, "29")) != ground:
        return False
    if not any(
        output["output_net"] == vdd and output["ground_net"] == ground
        and 3.0 <= output["voltage"] <= 3.6
        for output in observed_outputs
    ):
        return False
    upper = _resistor_path_value(vbus, sense, parts, pad_nets, 1)
    lower = _resistor_path_value(sense, ground, parts, pad_nets, 1)
    pullup = _resistor_path_value(vdd, reset, parts, pad_nets, 1)
    if not (
        len(upper) == len(lower) == len(pullup) == 1
        and upper[0][0] == 22100.0 and lower[0][0] == 47500.0
        and pullup[0][0] == 1000.0
    ):
        return False
    sense_owners = {ref, *upper[0][1], *lower[0][1]}
    reset_owners = {ref, *pullup[0][1]}
    if any(
        (net == sense and owner not in sense_owners)
        or (net == reset and owner not in reset_owners)
        for (owner, _pin), net in pad_nets.items()
    ):
        return False
    capacitors = [
        _capacitance_farads(str(capacitor.get("value") or ""))
        for capacitor, left, right in _two_terminal(parts, pad_nets, "C")
        if {left[1], right[1]} == {vdd, ground}
    ]
    return (
        sum(value is not None and value >= 4.7e-6 for value in capacitors) >= 2
        and sum(value is not None and math.isclose(value, 100e-9, rel_tol=0.2) for value in capacitors) >= 2
    )


def _reviewed_mcu_uart_nets(part, pad_nets) -> list[tuple[str, str]]:
    record = _record(part)
    if record is None or "microcontroller" not in record.physical_features:
        return []
    ref = str(part["ref"])
    pairs = []
    for tx_key, rx_key in (("txd0", "rxd0"), ("uart_tx", "uart_rx")):
        if tx_key in record.port_pins and rx_key in record.port_pins:
            pairs.append((record.port_pins[tx_key], record.port_pins[rx_key]))
    return [
        (tx, rx) for tx_pin, rx_pin in pairs
        if (tx := pad_nets.get((ref, str(tx_pin))))
        and (rx := pad_nets.get((ref, str(rx_pin))))
        and tx != rx
    ]


def _functional_facts(parts, names, pad_nets, state, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    from .mcu_artifact_evidence import (
        _CONNECTOR_IO_GATES, _CONNECTOR_PROTOCOL_GATES,
        _powered_complete_mcus,
        controller_function_gates, mcu_circuit_facts,
    )
    from .connector_artifact_evidence import connector_circuit_facts

    mcu_facts = mcu_circuit_facts(parts, names, pad_nets)
    complete_mcu_refs = set(mcu_facts.get("complete_mcu_refs", ()))
    usb_bridge_mcu_refs: set[str] = set()
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
    opamps = [
        p for p in parts
        if any(
            feature == "op-amp" or "operational-amplifier" in feature
            for feature in _physical_features(p)
        )
    ]
    # MAX31855: dedicated T+/T- plus CS/SCK/SO header evidence and actual supply/ground pins.
    for chip in _device(parts, "max31855"):
        tc = _pins_for(str(chip["ref"]), names, pad_nets, r"^(T\+|T-|TPLUS|TMINUS)$")
        spi = _pins_for(str(chip["ref"]), names, pad_nets, r"^(CS|SCK|SO|DO)$")
        powered = _one_pin(str(chip["ref"]), names, pad_nets, r"^(VCC|VDD)$") and _one_pin(str(chip["ref"]), names, pad_nets, r"^(GND|VSS)$")
        if len(tc) == 2 and len({n for _, n in tc}) == 2 and len(spi) >= 3 and powered:
            paths["spi_header"] = any(all(n in {x[1] for x in _all_pads(str(h["ref"]), pad_nets)} for _, n in spi) for h in connectors); paths["thermocouple_supply"] = bool(powered)
    # Voltage is published only for one complete converter path whose own
    # output reaches a real connector.  A reviewed IC current limit is not a
    # board output rating without independent source, thermal and support proof.
    dual_source = observe_dual_source_battery_power(
        parts, pad_nets,
        source_voltage=lambda _connector, positive, ground, kind, minimum, maximum: (
            _declared_dual_source_range(state, positive, ground, kind, minimum, maximum)
        ),
    )
    source_outputs = []
    if dual_source is not None:
        mux = next(part for part in parts if (_record(part) is not None and _record(part).identity == "tps2113apwr"))
        source_outputs.append({
            "ref": str(mux["ref"]), "input_net": pad_nets[str(mux["ref"]), "6"],
            "output_net": dual_source.output_net, "ground_net": dual_source.ground_net,
            "voltage": dual_source.output_voltage_v, "converting": False,
        })
    internal_outputs = _observed_power_outputs(parts, names, pad_nets, state, source_outputs)
    observed_outputs = _exposed_power_outputs(internal_outputs, parts, connectors, names, pad_nets)
    output_voltage = _select_output_voltage(
        [(entry["output_net"], entry["voltage"]) for entry in observed_outputs],
    )
    if output_voltage is not None:
        numeric["output_voltage_v"] = output_voltage
    # An IC's voltage ceiling is not a board input rating: other devices and
    # support components on the actual input path may impose a lower limit.
    for p in parts:
        record = _record(p)
        if record is None: continue
        bootstrap = record.bootstrap
        if bootstrap:
            boot = _one_pin(str(p["ref"]), names, pad_nets, "^" + re.escape(str(bootstrap.get("positive_pin"))) + "$"); sw = _one_pin(str(p["ref"]), names, pad_nets, "^" + re.escape(str(bootstrap.get("negative_pin"))) + "$")
            required = float(bootstrap.get("capacitance_uf", 0)) * 1e-6
            if boot and sw and boot[1] != sw[1] and any({a[1], b[1]} == {boot[1], sw[1]} and (v := _capacitance_farads(str(c.get("value") or ""))) is not None and abs(v-required) <= required*.2 for c,a,b in _two_terminal(parts,pad_nets,"C")):
                paths["switcher_support"] = True
        transfer = record.power_transfer
        if isinstance(transfer, dict):
            ref = str(p["ref"])
            from_pin = str(transfer.get("from_pin") or "")
            to_pin = str(transfer.get("to_pin") or "")
            source = (from_pin, pad_nets[(ref, from_pin)]) if (ref, from_pin) in pad_nets else _one_port_pin(ref, record, "input", names, pad_nets)
            dest = (to_pin, pad_nets[(ref, to_pin)]) if (ref, to_pin) in pad_nets else _one_port_pin(ref, record, "output", names, pad_nets)
            if source and dest and _reaches(graph, source[1], dest[1]):
                paths["reviewed_power_transfer"] = True
    # Follow only reviewed non-converting power paths to a complete LED and
    # valued series-resistor chain. A converter switch node is not a DC supply.
    input_pairs = _power_input_pairs(parts, names, pad_nets)
    unproved_switches = {
        str(part["ref"]) for part in parts
        if "p-channel-mosfet" in _physical_features(part)
        and not _enable_jumper_observed(parts, connectors, names, pad_nets, device_ref=str(part["ref"]))
        and not _reverse_pmos_observed(parts, connectors, input_pairs, names, pad_nets, device_ref=str(part["ref"]))
    }
    nominal_graph = _power_edges(
        [part for part in parts if str(part["ref"]) not in unproved_switches],
        names, pad_nets, nonconverting_only=True,
    )
    led_supplies = (supply_nets | {entry["output_net"] for entry in internal_outputs}) - ground_nets
    pending_supplies = list(led_supplies)
    for supply in pending_supplies:
        for peer in nominal_graph.get(supply, ()):
            if peer not in led_supplies and peer not in ground_nets:
                led_supplies.add(peer)
                pending_supplies.append(peer)
    led_channels = 0
    for led in [p for p in parts if _is_kind(p, "led") or "led-0805" in _reviewed_identity(p)]:
        a = _one_pin(str(led["ref"]), names, pad_nets, r"^(A|ANODE)$")
        k = _one_pin(str(led["ref"]), names, pad_nets, r"^(K|CATHODE)$")
        if not (a and k and a[1] != k[1]):
            continue
        source_path = (
            k[1] in ground_nets
            and any(len(_resistor_path_value(a[1], supply, parts, pad_nets, maximum=16)) == 1
                    for supply in led_supplies)
        )
        sink_path = (
            a[1] in led_supplies
            and any(len(_resistor_path_value(k[1], ground, parts, pad_nets, maximum=16)) == 1
                    for ground in ground_nets)
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
    addressable_rgb = _addressable_rgb_status_refs(
        parts, names, pad_nets, state, internal_outputs, complete_mcu_refs
    )
    ws = _device(parts, "ws2812b")
    if len(ws) >= 12 and {str(part["ref"]) for part in ws} == addressable_rgb:
        paths["ws2812b_data_chain"] = True
        counts["ws2812b"] = len(ws)
    if addressable_rgb:
        paths["status_led"] = True
        counts["status_led_circuit"] = counts.get("status_led_circuit", 0) + len(addressable_rgb)
    # External WS2812 strips are a connector contract, not an onboard pixel
    # count.  Prove a three-contact 5 V/DIN/GND topology and its real series
    # element from delivered pads; recipe names carry no authority here.
    for header in connectors:
        pads = dict(_all_pads(str(header["ref"]), pad_nets))
        if set(pads) != {"1", "2", "3"}:
            continue
        rail, data, ground = pads["1"], pads["2"], pads["3"]
        if not (_is_rail(rail) and _is_ground(ground) and data not in {rail, ground}):
            continue
        if any(
            _resistance_ohms(str(resistor.get("value") or "")) == 330.0
            and data in {left[1], right[1]}
            and (left[1] if right[1] == data else right[1]) not in {rail, ground}
            for resistor, left, right in _two_terminal(parts, pad_nets, "R")
        ):
            paths["ws2812_external_strip"] = True
            counts["ws2812_external_strip_output"] = 1
            break
    # Four separate common-cathode packages must land on the MAX7219's first
    # four DIG sinks.  A label or four cloned nets alone cannot create this fact.
    drivers = [p for p in parts if "seven-segment-driver" in _physical_features(p)]
    digits = [p for p in parts if "seven-segment-digit" in _physical_features(p)]
    for driver in drivers:
        digit_nets = [
            pad_nets.get((str(driver["ref"]), pin))
            for pin in ("2", "11", "6", "7")
        ]
        if len(set(digit_nets)) != 4 or any(net is None for net in digit_nets):
            continue
        attached = [
            digit for digit in digits
            if pad_nets.get((str(digit["ref"]), "3")) == pad_nets.get((str(digit["ref"]), "8"))
            and pad_nets.get((str(digit["ref"]), "3")) in digit_nets
        ]
        if len(attached) != 4:
            continue
        # At least A..G and DP of every digit must terminate at semantic MAX
        # segment pads; this rejects a display body with its digit channels only.
        segment_nets = {
            pad_nets.get((str(driver["ref"]), pin))
            for pin in ("14", "16", "20", "23", "21", "15", "17", "22")
        }
        level_ok = any(
            [pad_nets.get((str(level["ref"]), pin)) for pin in ("18", "17", "16")]
            == [pad_nets.get((str(driver["ref"]), pin)) for pin in ("1", "13", "12")]
            and len({
                pad_nets.get((str(level["ref"]), pin)) for pin in ("2", "3", "4")
            }) == 3
            and all(
                pad_nets.get((str(level["ref"]), pin)) not in {
                    pad_nets.get((str(level["ref"]), out)) for out in ("18", "17", "16")
                }
                for pin in ("2", "3", "4")
            )
            and pad_nets.get((str(level["ref"]), "20"))
            != pad_nets.get((str(level["ref"]), "10"))
            for level in parts
            if "logic-level-translator" in _physical_features(level)
        )
        if (
            level_ok
            and all(
                {
                    pad_nets.get((str(digit["ref"]), pin))
                    for pin in ("1", "2", "4", "5", "6", "7", "9", "10")
                } == segment_nets
                for digit in attached
            )
        ):
            paths["four_digit_seven_segment"] = True
            counts["seven_segment_digit"] = 4
            break
    # USB bridge evidence joins exact bridge, USB-C, ESD, both CC Rd networks,
    # protected D+/D- routes, and crossed UART terminals.  It deliberately
    # fails closed if a native MCU PHY is also on the bridge's pair.
    bridges = [p for p in parts if "usb-uart-bridge" in _physical_features(p)]
    usb_cs = [p for p in connectors if "usb-c-receptacle" in _physical_features(p)]
    usb_data_graph = _usb_data_graph(parts, pad_nets)
    for bridge in bridges:
        record = _record(bridge)
        if record is None or not {"usb_dp", "usb_dm", "txd", "rxd"} <= set(record.port_pins):
            continue
        ref = str(bridge["ref"])
        dp = pad_nets.get((ref, record.port_pins["usb_dp"]))
        dm = pad_nets.get((ref, record.port_pins["usb_dm"]))
        tx = pad_nets.get((ref, record.port_pins["txd"]))
        rx = pad_nets.get((ref, record.port_pins["rxd"]))
        vdd_pin = record.port_pins.get("vdd", record.port_pins.get("vcc"))
        vdd = pad_nets.get((ref, vdd_pin)) if vdd_pin else None
        ground = next(
            (net for _pin, net in _pins_for(ref, names, pad_nets, r"^(GND|VSS|PGND)$")),
            None,
        )
        if not all((dp, dm, tx, rx, vdd, ground)) or len({dp, dm, tx, rx, vdd, ground}) != 6:
            continue
        support_ok = any(
            {left[1], right[1]} == {vdd, ground}
            for _capacitor, left, right in _two_terminal(parts, pad_nets, "C")
        )
        same_phy = any(
            other is not bridge
            and "microcontroller" in _physical_features(other)
            and (other_record := _record(other)) is not None
            and {"usb_dp", "usb_dm"} <= set(other_record.port_pins)
            and any(
                pad_nets.get((str(other["ref"]), other_record.port_pins[port])) in {dp, dm}
                for port in ("usb_dp", "usb_dm")
            )
            for other in parts
        )
        if same_phy:
            continue
        attached_usb = [
            usb for usb in usb_cs
            if any(
                _reaches(usb_data_graph, signal, pad_nets.get((str(usb["ref"]), pin)))
                for signal in (dp, dm) for pin in ("A6", "B6", "A7", "B7")
            )
        ]
        if len(attached_usb) != 1:
            continue
        for usb in attached_usb:
            uref = str(usb["ref"])
            conn_dp = {pad_nets.get((uref, pin)) for pin in ("A6", "B6")}
            conn_dm = {pad_nets.get((uref, pin)) for pin in ("A7", "B7")}
            cc = [pad_nets.get((uref, pin)) for pin in ("A5", "B5")]
            bus_pins = [pad_nets.get((uref, pin)) for pin in ("A4B9", "B4A9")]
            if not bus_pins[0] or bus_pins[0] != bus_pins[1]:
                continue
            vbus = bus_pins[0]
            if any(
                other is not usb
                and any(pad_nets.get((str(other["ref"]), pin)) == vbus for pin in ("A4B9", "B4A9"))
                for other in usb_cs
            ):
                continue
            if (
                len(conn_dp) != 1 or len(conn_dm) != 1
                or None in conn_dp | conn_dm or conn_dp == conn_dm
                or not (conn_dp | conn_dm).isdisjoint({ground, vbus})
                or not all(cc) or len({*cc, dp, dm, tx, rx, vdd, ground}) != 8
                or vbus in {dp, dm, tx, rx, ground}
            ):
                continue
            if any(pad_nets.get((uref, pin)) != ground for pin in ("A1B12", "B1A12")):
                continue
            if record.identity == "cp2102n-a02-gqfn28" and not _cp2102n_support_observed(
                bridge, parts, pad_nets, vbus, vdd, ground, observed_outputs,
            ):
                continue
            if record.identity == "ch340c" and not _ch340c_support_observed(
                bridge, parts, pad_nets, vbus, vdd, ground, observed_outputs,
            ):
                continue
            cc_ok = all(
                net and any(
                    _resistance_ohms(str(resistor.get("value") or "")) == 5100.0
                    and net in {left[1], right[1]}
                    and (right[1] if left[1] == net else left[1]) == ground
                    for resistor, left, right in _two_terminal(parts, pad_nets, "R")
                )
                for net in cc
            )
            esd_rows = [
                esd for esd in parts
                if (esd_record := _record(esd)) is not None
                and (pairs := (esd_record.support_network or {}).get("usb_feed_through_pairs"))
                and pad_nets.get((str(esd["ref"]), esd_record.port_pins.get("ground"))) == ground
                and pad_nets.get((str(esd["ref"]), esd_record.port_pins.get("vbus"))) == vbus
                and any(
                    conn_dm == {pad_nets.get((str(esd["ref"]), pin))}
                    for pair in pairs for pin in pair
                )
                and any(
                    conn_dp == {pad_nets.get((str(esd["ref"]), pin))}
                    for pair in pairs for pin in pair
                )
            ]
            route_ok = (
                _reaches(usb_data_graph, dm, next(iter(conn_dm)))
                and _reaches(usb_data_graph, dp, next(iter(conn_dp)))
                and not _reaches(usb_data_graph, dm, dp)
            )
            esd_ok = bool(esd_rows)
            uart_mcus = {
                str(other["ref"])
                for other in parts if other is not bridge
                for other_tx, other_rx in _reviewed_mcu_uart_nets(other, pad_nets)
                if tx == other_rx and rx == other_tx and tx != rx
            }
            if cc_ok and esd_ok and route_ok and uart_mcus and support_ok:
                usb_bridge_mcu_refs.update(uart_mcus)
                paths["usb_uart_bridge"] = True
                paths["usb_c"] = True
                break
    # Audio follower facts are semantic OUT-to-negative-input connections for each channel.
    followers = 0
    for amp in opamps:
        outs = _pins_for(str(amp["ref"]), names, pad_nets, r"^(OUT|OUTPUT)[A-D]?$")
        negs = _pins_for(str(amp["ref"]), names, pad_nets, r"^(IN-|INV|-)IN[A-D]?$" )
        followers += sum(1 for _, out in outs if out in {n for _,n in negs})
    if followers: counts["audio_buffer_channel"] = followers; numeric["audio_headroom_v"] = 0.0
    # A current value and both topology paths are published together, only after
    # a complete reviewed loop survives actual-board reconciliation.
    current_a = _reviewed_led_current_loop(parts, names, pad_nets)
    if current_a is not None:
        numeric["led_current_a"] = current_a
        paths["current_feedback"] = True
        paths["switcher_support"] = True
        if any(
            "high-power-led" in _physical_features(part)
            and (_record(part) is not None)
            and float(_record(part).operating_limits.get("power_w", 0)) >= 3.0
            for part in parts
        ):
            paths["power_led_3w"] = True

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
    # A removable 120-ohm CAN termination is a series resistor/switch branch
    # between the actual CANH/CANL pads, not a claimed recipe role or net label.
    terminals = list(_two_terminal(parts, pad_nets))
    for transceiver in _device(parts, "sn65hvd230", "can transceiver"):
        high = _one_pin(str(transceiver["ref"]), names, pad_nets, r"CAN.?H|^H$")
        low = _one_pin(str(transceiver["ref"]), names, pad_nets, r"CAN.?L|^L$")
        if not high or not low or high[1] == low[1]:
            continue
        bus = {high[1], low[1]}
        terminators = [
            (part, left[1], right[1])
            for part, left, right in terminals
            if _is_kind(part, "resistor")
            and (resistance := _resistance_ohms(str(part.get("value") or ""))) is not None
            and 114.0 <= resistance <= 126.0
        ]
        if any({left, right} == bus for _, left, right in terminators):
            continue  # An unswitched parallel terminator cannot be disconnected.
        for resistor, left, right in terminators:
            branch = {left, right}
            if len(branch & bus) != 1 or len(branch) != 2:
                continue
            middle = next(iter(branch - bus))
            other_bus = next(iter(bus - branch))
            if any(
                switch["ref"] != resistor["ref"]
                and {a[1], b[1]} == {middle, other_bus}
                and (
                    _is_kind(switch, "jumper")
                    or bool({"switch", "jumper"} & _physical_features(switch))
                )
                for switch, a, b in terminals
            ):
                paths["switchable_can_termination"] = True
                break
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
    # MCU support is only published from the delivered pad graph, never from a
    # recipe selection or an intended controller family.
    paths.update(mcu_facts["net_paths"])
    for channel, count in mcu_facts["channel_counts"].items():
        counts[channel] = max(counts.get(channel, 0), count)
    lists["complete_mcu_refs"] = sorted(complete_mcu_refs)

    def supply_voltage(rail, ground):
        return _supply_voltage_observed(
            rail, ground, parts, names, pad_nets, state, internal_outputs
        )

    powered_mcu_refs = _powered_complete_mcus(parts, mcu_facts, supply_voltage)
    temperature_facts = temperature_acquisition_facts(
        parts, pad_nets, mcu_facts=mcu_facts,
        powered_mcu_refs=powered_mcu_refs, supply_voltage=supply_voltage,
    )
    analog_reset_observed = _analog_power_cycle_reset_observed(
        parts, names, pad_nets, state
    )
    def analog_supply_voltage(rail, ground):
        voltage = supply_voltage(rail, ground)
        if voltage is not None or not analog_reset_observed:
            return voltage
        # The reset observer proves the complete switched path. Restrict this
        # propagation to the single reviewed switch it can have proved.
        switches = _device(parts, "tps22948dckr")
        if len(switches) != 1:
            return None
        switch = switches[0]
        record = _record(switch)
        if record is None:
            return None
        ref = str(switch["ref"])
        source = _one_port_pin(ref, record, "input", names, pad_nets)
        output = _one_port_pin(ref, record, "output", names, pad_nets)
        reference = _one_port_pin(ref, record, "ground", names, pad_nets)
        if source and output and reference and output[1] == rail and reference[1] == ground:
            return supply_voltage(source[1], ground)
        return None

    analog_facts = analog_conditioner_facts(
        parts, names, pad_nets, state=state, reset_observed=analog_reset_observed,
        supply_voltage=analog_supply_voltage,
    )
    temt_facts = temt6000_readout_facts(
        parts, names, pad_nets, mcu_facts=mcu_facts,
        powered_mcu_refs=powered_mcu_refs, supply_voltage=supply_voltage,
    )
    for observed in (analog_facts, temt_facts, temperature_facts):
        paths.update(observed.get("net_paths", {}))
        for channel, count in observed.get("channel_counts", {}).items():
            counts[channel] = max(counts.get(channel, 0), count)
        numeric.update(
            {
                key: value
                for key, value in observed.items()
                if key not in {
                    "net_paths", "channel_counts", "gates", "analog_output_nets",
                    "ambient_light_acquisition_nets",
                }
            }
        )
    if "analog_output_nets" in analog_facts:
        lists["analog_output_nets"] = analog_facts["analog_output_nets"]
    if "ambient_light_acquisition_nets" in temt_facts:
        lists["ambient_light_acquisition_nets"] = temt_facts[
            "ambient_light_acquisition_nets"
        ]

    display_facts = display_circuit_facts(
        parts, names, pad_nets, mcu_facts=mcu_facts, supply_voltage=supply_voltage,
    )
    paths.update(display_facts["net_paths"])
    for channel, count in display_facts["channel_counts"].items():
        counts[channel] = max(counts.get(channel, 0), count)
    logger_facts = data_logging_hardware_facts(
        parts, names, pad_nets, mcu_facts=mcu_facts, analog_facts=analog_facts,
        powered_mcu_refs=powered_mcu_refs, supply_voltage=supply_voltage,
        independent_input_nets=temt_facts.get("ambient_light_acquisition_nets", ()),
    )
    paths.update(logger_facts["net_paths"])
    for channel, count in logger_facts["channel_counts"].items():
        counts[channel] = max(counts.get(channel, 0), count)
    for key in ("data_logging_storage_addresses", "data_logging_mcu_refs"):
        if key in logger_facts:
            lists[key] = logger_facts[key]
    power_stages = qualified_converter_power_stage_currents(
        parts, names, pad_nets, internal_outputs, supply_voltage,
    )
    if power_stages:
        # Component-level evidence is not board ampacity or thermal qualification.
        lists["converter_power_stages"] = power_stages

    actuator_facts = actuator_circuit_facts(
        parts, names, pad_nets, observed_outputs=internal_outputs,
        supply_voltage=supply_voltage, powered_mcu_refs=powered_mcu_refs,
    )
    paths.update(actuator_facts.get("net_paths", {}))
    for channel, count in actuator_facts.get("channel_counts", {}).items():
        counts[channel] = max(counts.get(channel, 0), count)
    numeric.update(
        {
            key: value
            for key, value in actuator_facts.items()
            if key not in {"net_paths", "channel_counts", "gates"}
        }
    )
    requested = {
        str(check.get("gate"))
        for obligation in (contract or {}).get("obligations", [])
        if isinstance(obligation, dict)
        and isinstance((check := obligation.get("check")), dict)
        and check.get("kind") == "gate"
    }
    connector_requests = requested
    if requested & {
        "external_development_board_function", "external_interface_board_function",
        "external_sensor_breakout_function",
    }:
        # Broader functions depend on these observations even if the original
        # contract did not also request a separate connector-specific verdict.
        connector_requests = requested | _CONNECTOR_PROTOCOL_GATES | _CONNECTOR_IO_GATES
    connector_facts = connector_circuit_facts(
        parts, names, pad_nets, requested=connector_requests, state=state,
        outputs=internal_outputs, complete_mcu_refs=complete_mcu_refs,
        powered_mcu_refs=powered_mcu_refs,
        complete_mcu_gpio_nets=mcu_facts.get("complete_mcu_gpio_nets"),
        complete_mcu_supplies=mcu_facts.get("complete_mcu_supplies"),
    )
    paths.update(connector_facts["net_paths"])
    for channel, count in connector_facts["channel_counts"].items():
        counts[channel] = max(counts.get(channel, 0), count)
    controller_gates = controller_function_gates(
        parts, names, pad_nets, state=state, mcu_facts=mcu_facts,
        connector_facts=connector_facts, analog_facts=analog_facts,
        outputs=internal_outputs, supply_voltage=supply_voltage,
    )
    gates = _power_control_gates(
        parts, connectors, names, pad_nets, requested, state, observed_outputs, internal_outputs,
    )
    for observed_gates in (
        analog_facts.get("gates", {}), temt_facts.get("gates", {}),
        temperature_facts["gates"],
        actuator_facts.get("gates", {}), connector_facts["gates"], controller_gates,
        display_facts["gates"], logger_facts["gates"],
    ):
        gates.update({key: value for key, value in observed_gates.items() if key in requested})
    if dual_source is not None:
        battery_supplied = {dual_source.output_net}
        for _ in range(len(internal_outputs)):
            expanded = battery_supplied | {
                row["output_net"] for row in internal_outputs
                if row["input_net"] in battery_supplied and row["ground_net"] == dual_source.ground_net
            }
            if expanded == battery_supplied:
                break
            battery_supplied = expanded
        supplies = mcu_facts.get("complete_mcu_supplies", {})
        if any(
            supplies.get(ref, {}).get("rail") in battery_supplied
            and supplies.get(ref, {}).get("ground") == dual_source.ground_net
            for ref in _powered_complete_mcus(parts, mcu_facts, supply_voltage)
        ):
            paths["dual_source_battery_to_controller"] = True
            for gate in ("external_battery_powered_operation", "external_enable_jumper_function"):
                if gate in requested:
                    gates[gate] = "pass"
    if (
        "external_usb_serial_bridge_function" in requested
        and paths.get("usb_uart_bridge")
        and usb_bridge_mcu_refs & _powered_complete_mcus(parts, mcu_facts, supply_voltage)
    ):
        gates["external_usb_serial_bridge_function"] = "pass"
    if "external_four_pin_jst_connector" in requested and any(
        (record := _record(connector)) is not None
        and {"jst-connector", "jst-ph", "jst-xh"} & record.physical_features
        and len(record.contacts) == 4
        and len(pads := _all_pads(str(connector["ref"]), pad_nets)) == 4
        and len({net for _, net in pads}) >= 2
        and all(
            any(other_ref != str(connector["ref"]) and other_net == net
                for (other_ref, _), other_net in pad_nets.items())
            for _, net in pads
        )
        for connector in connectors
    ):
        gates["external_four_pin_jst_connector"] = "pass"
    if "external_enable_jumper_function" in requested and _enable_jumper_observed(parts, connectors, names, pad_nets):
        gates["external_enable_jumper_function"] = "pass"
    if "external_reset_control_function" in requested and (
        _reset_button_observed(parts, names, pad_nets, powered_mcu_refs) or analog_reset_observed
    ):
        gates["external_reset_control_function"] = "pass"
    if "external_rgb_status_led_channels" in requested and (
        display_facts["gates"].get("external_rgb_status_led_channels") == "pass"
        or addressable_rgb
        or _rgb_status_observed(
            parts,
            *_rgb_power_context(parts, connectors, names, pad_nets, state, observed_outputs, nominal_graph),
            names, pad_nets,
        )
    ):
        gates["external_rgb_status_led_channels"] = "pass"
    return {"net_paths": paths, "channel_counts": counts, **numeric, **lists, "gates": gates}


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
    functional = _functional_facts(parts, names, pad_nets, state, contract)
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
    # The native lowerer witness proves the whole R-2R topology, not merely a
    # population of R/2R values. Reconciliation above binds that canonical BOM
    # graph to the delivered pads before the witness can become board evidence.
    if bom is not None and connections_complete:
        try:
            from kicraft.design.models import Architecture
            from kicraft.design.synthesis.validation import _canonical_lowerer_refs

            architecture = Architecture.model_validate(state.get("architecture"))
            ladders = [
                requirement for requirement in architecture.requirements
                if len(refs := _canonical_lowerer_refs(
                    requirement, bom, component_class="resistor-network"
                )) >= 16
                and all(
                    part.resolution_id == "r2r-ladder@1"
                    for part in bom.parts if part.ref in refs
                )
            ]
            if ladders:
                functional["net_paths"]["r2r_ladder"] = True
                connector_nets = {
                    net for connector in _connector(parts)
                    for _, net in _all_pads(str(connector["ref"]), pad_nets)
                }
                if any(
                    len(bits := {requirement.ports.get(f"bit{i}") for i in range(8)}) == 8
                    and None not in bits and bits <= connector_nets
                    for requirement in ladders
                ):
                    functional["channel_counts"]["logic_input"] = 8
                for part in parts:
                    if not any(
                        feature == "op-amp" or "operational-amplifier" in feature
                        for feature in _physical_features(part)
                    ):
                        continue
                    output = _one_pin(str(part["ref"]), names, pad_nets, r"^(OUT|OUTPUT)$")
                    inverted = _one_pin(str(part["ref"]), names, pad_nets, r"^(IN-|INV|-)IN$")
                    if output and inverted and output[1] == inverted[1]:
                        functional["net_paths"]["analog_output_buffer"] = True
                        break
        except (TypeError, ValueError, AttributeError):
            pass
    # Programming is an established BOM semantic check, but its BOM-only result
    # cannot outlive a missing, netless, or substituted delivered endpoint.
    try:
        from kicraft.design.synthesis.validation import mcu_programming_facts

        if bom is not None and connections_complete:
            program = mcu_programming_facts(bom, architecture=state.get("architecture"))
            if program and program["access_ok"] and program["path_ok"]:
                functional.setdefault("gates", {})["programming"] = "pass"
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
