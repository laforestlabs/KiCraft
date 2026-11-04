"""Fail-closed external-connector evidence from delivered footprint pads.

This module deliberately does not use BOM quantities, net names, or recipe roles as
connector evidence.  A contract selects which observer is relevant; every positive
claim below is based on reviewed physical contacts and their delivered board nets.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.validation import _resistance_ohms
from kicraft.eval.application_pin_evidence import reviewed_application_pins
from kicraft.eval.electrical_artifact_evidence import _supply_voltage_observed


_GROUND_PORTS = frozenset({"ground", "gnd", "vss", "return", "0v"})
_SUPPLY_PORTS = frozenset({"vcc", "vdd", "vplus", "supply", "power"})
_PROTOCOLS = {
    "uart": {
        "tx": (("uart_tx", "txd", "txd0", "tx"), "uart-tx"),
        "rx": (("uart_rx", "rxd", "rxd0", "rx"), "uart-rx"),
    },
    "i2c": {
        "sda": (("sda", "i2c_sda"), "i2c-sda"),
        "scl": (("scl", "i2c_scl"), "i2c-scl"),
    },
    "spi": {
        "sclk": (("spi_sclk", "sclk", "sck"), "spi-sclk"),
        "mosi": (("spi_mosi", "mosi"), "spi-mosi"),
        "miso": (("spi_miso", "miso", "so"), "spi-miso"),
        "cs": (("spi_cs0", "spi_cs", "chip_select", "nss", "cs"), "spi-cs"),
    },
}


def _record(part: Mapping[str, Any]):
    return physical_inventory_record(
        mpn=part.get("mpn"),
        symbol=part.get("symbol"),
        footprint=part.get("footprint"),
        datasheet=part.get("datasheet"),
        sourcing_note=part.get("sourcing_note"),
    )


def _part_ref(part: Mapping[str, Any]) -> str:
    return str(part.get("ref") or "")


def _native_pads(
    part: Mapping[str, Any], record: Any, pad_nets: Mapping[tuple[str, str], str],
) -> dict[str, str] | None:
    """Return complete reviewed contacts only when each has a real board net."""
    ref = _part_ref(part)
    contacts = tuple(str(contact) for contact in record.contacts)
    if not ref or not contacts:
        return None
    pads = {
        pin: net
        for (candidate_ref, pin), net in pad_nets.items()
        if candidate_ref == ref and pin in contacts and isinstance(net, str) and net
    }
    if set(pads) != set(contacts):
        return None
    # Contact count is physical, not a count of independent signals. Shared
    # returns or duplicated supply contacts are legitimate connector pinouts.
    if len(set(pads.values())) < 2:
        return None
    return pads


def _lands_on_other_pad(
    ref: str, nets: Iterable[str], pad_nets: Mapping[tuple[str, str], str],
) -> bool:
    return all(
        any(other_ref != ref and other_net == net for (other_ref, _), other_net in pad_nets.items())
        for net in nets
    )


def _connected_native_connector(
    part: Mapping[str, Any], record: Any, pad_nets: Mapping[tuple[str, str], str],
) -> dict[str, str] | None:
    pads = _native_pads(part, record, pad_nets)
    if pads is None or not _lands_on_other_pad(_part_ref(part), pads.values(), pad_nets):
        return None
    return pads


def _is_screw_terminal(record: Any) -> bool:
    if not ({"screw-terminal", "terminal-block"} & set(record.physical_features)):
        return False
    pitch = record.operating_limits.get("pitch_mm")
    return pitch == 5.0 or "5.00 mm-pitch" in record.package


def _is_xh(record: Any) -> bool:
    return (
        "jst-xh-connector" in record.physical_features
        and record.family == "jst-xh-connector"
        and record.operating_limits.get("pitch_mm") == 2.5
    )


def _is_0p1_header(record: Any) -> bool:
    if not ({"pin-header", "header"} & set(record.physical_features)):
        return False
    pitch = record.operating_limits.get("pitch_mm")
    return pitch == 2.54 or "2.54 mm" in record.package


def _power_pins(record: Any) -> tuple[set[str], set[str]]:
    ports = record.port_pins or {}
    grounds = {str(pin) for name, pin in ports.items() if name.casefold() in _GROUND_PORTS}
    supplies = {str(pin) for name, pin in ports.items() if name.casefold() in _SUPPLY_PORTS}
    return grounds, supplies


def _protocol_feature_allowed(record: Any, protocol: str) -> bool:
    features = set(record.physical_features)
    if "microcontroller" in features:
        return True
    if protocol == "uart":
        return any("uart" in feature for feature in features)
    return any(protocol in feature for feature in features)


def _protocol_endpoints(
    parts: Iterable[Mapping[str, Any]], pad_nets: Mapping[tuple[str, str], str], protocol: str,
    complete_mcu_refs: set[str],
) -> list[dict[str, Any]]:
    """Return actual reviewed device pins capable of every protocol conductor.

    Names in a netlist are intentionally ignored: either a reviewed fixed port or
    a production recipe's allocatable capability must identify each physical pad.
    """
    capabilities = {capability: reviewed_application_pins(capability) for _, capability in _PROTOCOLS[protocol].values()}
    endpoints: list[dict[str, Any]] = []
    for part in parts:
        record = _record(part)
        ref = _part_ref(part)
        if (
            record is None
            or not ref
            or not _protocol_feature_allowed(record, protocol)
            or ("microcontroller" in record.physical_features and ref not in complete_mcu_refs)
        ):
            continue
        grounded, supplied = _power_pins(record)
        ground_nets = {pad_nets.get((ref, pin)) for pin in grounded} - {None, ""}
        supply_nets = {pad_nets.get((ref, pin)) for pin in supplied} - {None, ""}
        if len(ground_nets) != 1 or len(supply_nets) != 1:
            continue
        pins_by_role: dict[str, set[str]] = {}
        for role, (port_names, capability) in _PROTOCOLS[protocol].items():
            pins = {
                str(record.port_pins[name])
                for name in port_names
                if name in record.port_pins
            }
            if "microcontroller" in record.physical_features:
                pins.update(capabilities[capability].get(record.identity, set()))
            nets = {
                pad_nets[(ref, pin)]
                for pin in pins
                if (ref, pin) in pad_nets and pad_nets[(ref, pin)]
            }
            if not nets:
                break
            pins_by_role[role] = nets
        else:
            endpoints.append(
                {
                    "ref": ref,
                    "nets": pins_by_role,
                    "ground": next(iter(ground_nets)),
                    "supply": next(iter(supply_nets)),
                }
            )
    return endpoints


def _distinct_protocol_nets(
    nets_by_role: Mapping[str, set[str]], available: set[str],
) -> bool:
    """Find a different delivered header net for every typed conductor."""
    options = [
        tuple(sorted(nets & available))
        for nets in nets_by_role.values()
    ]
    if any(not option for option in options):
        return False

    def assign(index: int, used: set[str]) -> bool:
        if index == len(options):
            return True
        return any(
            net not in used and assign(index + 1, used | {net})
            for net in options[index]
        )

    return assign(0, set())


def _header_binds_protocol(
    headers: Iterable[tuple[Mapping[str, Any], dict[str, str]]], endpoints: Iterable[Mapping[str, Any]],
) -> bool:
    for _header, header_pads in headers:
        header_nets = set(header_pads.values())
        for endpoint in endpoints:
            if (
                endpoint["ground"] in header_nets
                and endpoint["supply"] in header_nets
                and endpoint["ground"] != endpoint["supply"]
                and _distinct_protocol_nets(endpoint["nets"], header_nets - {
                    endpoint["ground"], endpoint["supply"],
                })
            ):
                return True
    return False


def _qwiic_binds_i2c(
    parts: Iterable[Mapping[str, Any]], pad_nets: Mapping[tuple[str, str], str],
    endpoints: Iterable[Mapping[str, Any]], state: Mapping[str, Any],
    names: Mapping[tuple[str, str], str], outputs: Iterable[Mapping[str, Any]],
) -> bool:
    for part in parts:
        record = _record(part)
        if record is None or "qwiic-connector" not in record.physical_features:
            continue
        # This fixed Qwiic/STEMMA QT order is the published pin assignment, not
        # a generic four-contact JST that happens to carry I2C-named nets.
        if {key: str(value) for key, value in record.port_pins.items()} != {
            "ground": "1", "vcc": "2", "sda": "3", "scl": "4"
        }:
            continue
        pads = _connected_native_connector(part, record, pad_nets)
        if pads is None:
            continue
        ground, vcc, sda, scl = (pads[pin] for pin in ("1", "2", "3", "4"))
        for endpoint in endpoints:
            if not (
                sda in endpoint["nets"]["sda"]
                and scl in endpoint["nets"]["scl"]
                and sda != scl
            ):
                continue
            if endpoint["ground"] != ground or endpoint["supply"] != vcc:
                continue
            voltage = _supply_voltage_observed(
                vcc, ground, parts, names, pad_nets, state, outputs
            )
            if (
                isinstance(voltage, (int, float))
                and not isinstance(voltage, bool)
                and 3.25 <= float(voltage) <= 3.35
            ):
                return True
    return False


def _is_resistor(part: Mapping[str, Any]) -> bool:
    record = _record(part)
    return record is not None and "resistor" in record.family


def _pulls_to_supply(
    parts: Iterable[Mapping[str, Any]], pad_nets: Mapping[tuple[str, str], str], control: str, supply: str,
) -> bool:
    for part in parts:
        if not _is_resistor(part):
            continue
        ref = _part_ref(part)
        pins = [net for (candidate_ref, _), net in pad_nets.items() if candidate_ref == ref and net]
        if len(pins) != 2 or set(pins) != {control, supply}:
            continue
        resistance = _resistance_ohms(str(part.get("value") or ""))
        if resistance is not None and resistance >= 1_000.0:
            return True
    return False


def _three_button_functions(
    parts: list[Mapping[str, Any]], pad_nets: Mapping[tuple[str, str], str],
    powered_mcu_refs: set[str],
) -> bool:
    input_pins = reviewed_application_pins("input")
    controls: set[str] = set()
    for button in parts:
        record = _record(button)
        if record is None or "momentary-button" not in record.physical_features:
            continue
        pads = _connected_native_connector(button, record, pad_nets)
        if pads is None or len(pads) != 2:
            continue
        button_nets = set(pads.values())
        for target in parts:
            target_record = _record(target)
            target_ref = _part_ref(target)
            if (
                target_record is None
                or "microcontroller" not in target_record.physical_features
                or not target_ref
                or target_ref not in powered_mcu_refs
            ):
                continue
            grounds, supplies = _power_pins(target_record)
            ground_nets = {pad_nets.get((target_ref, pin)) for pin in grounds} - {None, ""}
            supply_nets = {pad_nets.get((target_ref, pin)) for pin in supplies} - {None, ""}
            if len(ground_nets) != 1 or len(supply_nets) != 1:
                continue
            functional_pins = set(input_pins.get(target_record.identity, set()))
            # Dedicated reset/enable and manual boot controls are functional
            # push buttons too; they need not be application GPIOs.
            functional_pins.update(
                target_record.port_pins[port]
                for port in ("reset", "enable", "boot", "io0", "io9")
                if port in target_record.port_pins
            )
            input_nets = {
                pad_nets.get((target_ref, pin))
                for pin in functional_pins
            } - {None, ""}
            candidates = button_nets & input_nets
            returns = button_nets & ground_nets
            if len(candidates) != 1 or len(returns) != 1:
                continue
            control = next(iter(candidates))
            supply = next(iter(supply_nets))
            if control != next(iter(returns)) and _pulls_to_supply(parts, pad_nets, control, supply):
                controls.add(control)
                break
    return len(controls) >= 3


def _powered_mcu_io_domains(
    powered_mcu_refs: Iterable[str],
    complete_mcu_gpio_nets: Mapping[str, Iterable[str]] | None,
    complete_mcu_supplies: Mapping[str, Mapping[str, str]] | None,
) -> list[tuple[set[str], str, str]]:
    """Return only board-backed GPIO/supply domains of powered complete MCUs."""
    gpio_nets = complete_mcu_gpio_nets or {}
    supplies = complete_mcu_supplies or {}
    domains: list[tuple[set[str], str, str]] = []
    for ref in {str(candidate) for candidate in powered_mcu_refs}:
        supply = supplies.get(ref)
        if not isinstance(supply, Mapping):
            continue
        rail, ground = supply.get("rail"), supply.get("ground")
        if not isinstance(rail, str) or not isinstance(ground, str) or not rail or not ground or rail == ground:
            continue
        owned_gpio_nets = {
            net
            for net, sources in gpio_nets.items()
            if isinstance(net, str)
            and net not in {rail, ground}
            and isinstance(sources, Iterable)
            and any(
                isinstance(source, str)
                and source.rpartition(".")[0] == ref
                for source in sources
            )
        }
        if owned_gpio_nets:
            domains.append((owned_gpio_nets, rail, ground))
    return domains


def _screw_terminal_group_binds_powered_mcu_io(
    pads: Mapping[str, str],
    domains: Iterable[tuple[set[str], str, str]],
) -> bool:
    """Require every terminal contact to belong to one powered MCU domain."""
    terminal_nets = set(pads.values())
    for gpio_nets, rail, ground in domains:
        if terminal_nets & gpio_nets and terminal_nets <= gpio_nets | {rail, ground}:
            return True
    return False


def connector_circuit_facts(
    parts: list[dict[str, Any]],
    names: Mapping[tuple[str, str], str],
    pad_nets: Mapping[tuple[str, str], str],
    *,
    requested: set[str],
    state: Mapping[str, Any],
    outputs: Iterable[Mapping[str, Any]],
    complete_mcu_refs: Iterable[str],
    powered_mcu_refs: Iterable[str] = (),
    complete_mcu_gpio_nets: Mapping[str, Iterable[str]] | None = None,
    complete_mcu_supplies: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Return requested connector gates proven by reviewed delivered pad contacts.

    ``names`` only reaches the shared supply observer; it never establishes an
    external connector, protocol contact, or board function by itself.
    """
    complete_mcu_refs = {str(ref) for ref in complete_mcu_refs}
    gates: dict[str, str] = {}
    connected: list[tuple[Mapping[str, Any], Any, dict[str, str]]] = []
    for part in parts:
        record = _record(part)
        if record is None:
            continue
        pads = _connected_native_connector(part, record, pad_nets)
        if pads is not None:
            connected.append((part, record, pads))

    terminals: list[dict[str, str]] = []
    if requested & {
        "external_four_screw_terminal_contacts",
        "external_four_screw_terminal_io_contacts",
    }:
        terminals = [
            pads
            for _part, record, pads in connected
            if _is_screw_terminal(record)
        ]
    if "external_four_screw_terminal_contacts" in requested and any(
        sum(len(pads) for pads in group) == 4
        for size in range(1, len(terminals) + 1)
        for group in combinations(terminals, size)
    ):
        gates["external_four_screw_terminal_contacts"] = "pass"

    if "external_four_screw_terminal_io_contacts" in requested:
        # This gate measures contacts, not distinct channels: two reviewed
        # signal/return terminals are four valid contacts when both signals
        # belong to powered complete-controller domains.
        domains = _powered_mcu_io_domains(
            powered_mcu_refs, complete_mcu_gpio_nets, complete_mcu_supplies,
        )
        if any(
            all(_screw_terminal_group_binds_powered_mcu_io(pads, domains) for pads in group)
            for size in range(1, len(terminals) + 1)
            for group in combinations(terminals, size)
            if sum(len(pads) for pads in group) == 4
        ):
            gates["external_four_screw_terminal_io_contacts"] = "pass"

    if "external_two_jst_xh_connectors" in requested:
        if sum(1 for _part, record, _pads in connected if _is_xh(record)) >= 2:
            gates["external_two_jst_xh_connectors"] = "pass"

    for contact_count, gate in (
        (8, "external_eight_pin_0p1_header"),
        (6, "external_six_pin_0p1_header"),
    ):
        if gate in requested and any(
            _is_0p1_header(record) and len(pads) == contact_count
            for _part, record, pads in connected
        ):
            gates[gate] = "pass"

    headers = [
        (part, pads)
        for part, record, pads in connected
        if _is_0p1_header(record)
    ]
    protocol_gates = {
        "external_uart_header_binding": ("uart", 4),
        "external_i2c_header_binding": ("i2c", 4),
        "external_spi_header_binding": ("spi", 6),
    }
    for gate, (protocol, minimum_contacts) in protocol_gates.items():
        if gate not in requested:
            continue
        compatible_headers = [(part, pads) for part, pads in headers if len(pads) >= minimum_contacts]
        if compatible_headers and _header_binds_protocol(
            compatible_headers, _protocol_endpoints(
                parts, pad_nets, protocol, complete_mcu_refs
            )
        ):
            gates[gate] = "pass"

    if "external_qwiic_i2c_connector_binding" in requested and _qwiic_binds_i2c(
        parts, pad_nets, _protocol_endpoints(parts, pad_nets, "i2c", complete_mcu_refs),
        state, names, outputs,
    ):
        gates["external_qwiic_i2c_connector_binding"] = "pass"

    if "external_three_push_button_functions" in requested and _three_button_functions(
        parts, pad_nets, set(powered_mcu_refs)
    ):
        gates["external_three_push_button_functions"] = "pass"

    return {"gates": gates, "net_paths": {}, "channel_counts": {}}
