"""Stable capability-based allocation for recipe-owned MCU application pins."""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from kicraft.design.models import CircuitRequirement, RecipePinAllocation

from .models import RecipeAllocatablePin, RecipeDefinition


class PinAllocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    net: str
    capability: str
    group: str | None = None
    contiguous: bool = False
    allow_strapping: bool = False


class PinAllocationError(ValueError):
    """Concrete deterministic failure; callers must not fall back to a model."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        requirement_id: str | None = None,
        capability: str | None = None,
        count: int | None = None,
        evidence: list[str] | None = None,
    ):
        self.code = code
        self.requirement_id = requirement_id
        self.capability = capability
        self.count = count
        self.evidence = evidence or []
        super().__init__(f"{code}: {message}")

# Exact logical keys, in canonical-first order. Metadata and allocation share
# this vocabulary; device-fixed ports are not additional application pins.
FIXED_INTERFACES = {
    "i2c_controller": (
        (("sda", "i2c_sda"), "i2c-sda"),
        (("scl", "i2c_scl"), "i2c-scl"),
    ),
    "spi_controller": (
        (("sclk", "spi_sclk"), "spi-sclk"),
        (("mosi", "spi_mosi"), "spi-mosi"),
        (("miso", "spi_miso"), "spi-miso"),
        (("cs", "spi_cs"), "spi-cs"),
    ),
    "uart": ((("tx", "uart_tx"), "uart-tx"), (("rx", "uart_rx"), "uart-rx")),
    "can_controller": ((("can_tx",), "can-tx"), (("can_rx",), "can-rx")),
}


def _interface_port_error(
    requirement: CircuitRequirement,
    interface: str,
    required: list[str],
    *,
    conflict: bool = False,
) -> PinAllocationError:
    evidence = [
        f"required_keys={required!r}",
        f"actual_bindings={dict(sorted(requirement.ports.items()))!r}",
    ]
    return PinAllocationError(
        "conflicting_interface_port" if conflict else "missing_interface_port",
        f"{interface}: {'conflicting aliases' if conflict else 'missing required binding'}; "
        + "; ".join(evidence),
        requirement_id=requirement.id,
        evidence=evidence,
    )


def _capability(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _supports(pin: RecipeAllocatablePin, request: PinAllocationRequest) -> bool:
    wanted = _capability(request.capability)
    available = {_capability(value) for value in pin.capabilities}
    if pin.reserved or (pin.strapping and not request.allow_strapping):
        return False
    if pin.input_only and wanted in {
        "gpio",
        "output",
        "pwm",
        "i2c-scl",
        "i2c-sda",
        "spi-sclk",
        "spi-mosi",
        "spi-cs",
        "uart-tx",
        "can-tx",
    }:
        return False
    return wanted in available or (wanted in {"input", "output"} and "gpio" in available)


def allocatable_capability_counts(pins: tuple[RecipeAllocatablePin, ...]) -> dict[str, int]:
    """Count legal default choices per capability, not jointly assignable capacity."""
    capabilities = {_capability(value) for pin in pins for value in pin.capabilities}
    if "gpio" in capabilities:
        capabilities.update(("input", "output"))
    counts = {}
    for capability in sorted(capabilities):
        request = PinAllocationRequest(id=capability, net=capability, capability=capability)
        count = len({pin.pin for pin in pins if _supports(pin, request)})
        if count:
            counts[capability] = count
    return counts


def _pin_key(pin: RecipeAllocatablePin) -> tuple[int, int, str]:
    return (
        1 if pin.strapping else 0,
        pin.gpio if pin.gpio is not None else 10_000,
        pin.pin,
    )


def _existing_by_net(
    existing: list[RecipePinAllocation] | None,
) -> dict[str, RecipePinAllocation]:
    return {allocation.net: allocation for allocation in (existing or [])}


def _request_key(request: PinAllocationRequest) -> tuple[str, int, str]:
    stem, separator, suffix = request.id.rpartition("_")
    if separator and suffix.isdigit():
        return stem, int(suffix), request.id
    return request.id, -1, request.id


def allocate_pins(
    pins: tuple[RecipeAllocatablePin, ...],
    requests: list[PinAllocationRequest],
    *,
    existing: list[RecipePinAllocation] | None = None,
    requirement_id: str | None = None,
) -> list[RecipePinAllocation]:
    """Allocate atomic request groups with stable ordering and tie-breaking."""
    if len({request.id for request in requests}) != len(requests):
        raise PinAllocationError(
            "duplicate_pin_request",
            "pin allocation request ids must be unique",
            requirement_id=requirement_id,
        )
    if len({request.net for request in requests}) != len(requests):
        raise PinAllocationError(
            "conflicting_pin_request",
            "each application net needs one MCU pin request; remove conflicting capability aliases",
            requirement_id=requirement_id,
        )
    existing_map = _existing_by_net(existing)
    pin_by_number = {pin.pin: pin for pin in pins}
    used: set[str] = set()
    result: list[RecipePinAllocation] = []
    groups: dict[str, list[PinAllocationRequest]] = defaultdict(list)
    for request in requests:
        groups[request.group or request.id].append(request)

    ordered_pins = sorted(pins, key=_pin_key)
    compatible = {
        request.id: frozenset(pin.pin for pin in ordered_pins if _supports(pin, request))
        for request in requests
    }
    by_gpio = {pin.gpio: pin for pin in ordered_pins if pin.gpio is not None}
    contiguous_windows: dict[str, list[list[RecipeAllocatablePin]]] = {}
    for group_name, group in groups.items():
        group.sort(key=_request_key)
        if not any(request.contiguous for request in group):
            continue
        windows = []
        for start in sorted(by_gpio):
            window = [by_gpio.get(start + offset) for offset in range(len(group))]
            if all(
                pin is not None and pin.pin in compatible[request.id]
                for pin, request in zip(window, group, strict=True)
            ):
                windows.append([pin for pin in window if pin is not None])
        contiguous_windows[group_name] = windows

    # Reserve valid existing groups before any new request can consume their pins.
    pending: dict[str, list[PinAllocationRequest]] = {}
    for group_name in sorted(groups):
        group = groups[group_name]
        retained: list[RecipePinAllocation] = []
        retained_pins: set[str] = set()
        for request in group:
            prior = existing_map.get(request.net)
            pin = pin_by_number.get(prior.pin) if prior is not None else None
            if (
                prior is None
                or pin is None
                or prior.pin in used
                or prior.pin in retained_pins
                or pin.pin not in compatible[request.id]
            ):
                retained = []
                break
            retained.append(
                RecipePinAllocation(
                    net=request.net,
                    pin=prior.pin,
                    capability=request.capability,
                )
            )
            retained_pins.add(prior.pin)
        if retained and len(retained) == len(group):
            if any(request.contiguous for request in group):
                gpios = [pin_by_number[allocation.pin].gpio for allocation in retained]
                concrete_gpios = sorted(gpio for gpio in gpios if gpio is not None)
                if len(concrete_gpios) != len(gpios) or concrete_gpios != list(
                    range(concrete_gpios[0], concrete_gpios[0] + len(gpios))
                ):
                    retained = []
            if retained:
                used.update(retained_pins)
                result.extend(retained)
                continue
        pending[group_name] = group

    while pending:
        # Scarcity is the number of legal choices, not a capability-name priority.
        # A contiguous bus has windows as choices; its members must stay atomic.
        group_name = min(
            pending,
            key=lambda name: (
                sum(
                    all(pin.pin not in used for pin in window)
                    for window in contiguous_windows[name]
                )
                if name in contiguous_windows
                else min(
                    sum(pin not in used for pin in compatible[request.id])
                    for request in pending[name]
                ),
                -len(pending[name]),
                name,
            ),
        )
        group = pending.pop(group_name)

        candidates = [pin for pin in ordered_pins if pin.pin not in used]
        chosen: list[RecipeAllocatablePin] = []
        if group_name in contiguous_windows:
            chosen = next(
                (
                    window
                    for window in contiguous_windows[group_name]
                    if all(pin.pin not in used for pin in window)
                ),
                [],
            )
        else:
            group = sorted(
                group,
                key=lambda request: (
                    sum(pin not in used for pin in compatible[request.id]),
                    _request_key(request),
                ),
            )
            remaining = list(candidates)
            for request in group:
                match = next(
                    (pin for pin in remaining if pin.pin in compatible[request.id]),
                    None,
                )
                if match is None:
                    break
                chosen.append(match)
                remaining.remove(match)
        if len(chosen) != len(group):
            missing = (
                group[len(chosen)].capability if len(chosen) < len(group) else group[0].capability
            )
            raise PinAllocationError(
                "unsatisfied_pin_capability",
                f"group {group_name!r} needs {len(group)} legal pins; missing {missing!r}",
                requirement_id=requirement_id,
                capability=missing,
                count=len(group),
            )
        used.update(pin.pin for pin in chosen)
        result.extend(
            RecipePinAllocation(
                net=request.net,
                pin=pin.pin,
                capability=request.capability,
            )
            for request, pin in zip(group, chosen, strict=True)
        )
    return sorted(result, key=lambda allocation: (allocation.net, allocation.pin))


def requests_from_requirement(
    requirement: CircuitRequirement,
    *,
    fixed_ports: set[str] | frozenset[str] = frozenset(),
) -> list[PinAllocationRequest]:
    """Translate bounded architecture interfaces into exact capability requests."""
    requests: list[PinAllocationRequest] = []
    interfaces = set(requirement.interfaces)
    if "parallel_output" in interfaces:
        count = requirement.parameters.get("parallel_output_count")
        if not isinstance(count, int) or not 1 <= count <= 32:
            raise PinAllocationError(
                "invalid_parallel_output_count",
                "parallel_output_count must be an integer from 1 through 32",
                requirement_id=requirement.id,
            )
        for index in range(count):
            port = f"parallel_{index}"
            net = requirement.ports.get(port)
            if not net:
                raise _interface_port_error(
                    requirement, "parallel_output", [f"parallel_{i}" for i in range(count)]
                )
            requests.append(
                PinAllocationRequest(
                    id=port,
                    net=net,
                    capability="output",
                    group="parallel_output",
                    contiguous=True,
                )
            )
    for interface, members in FIXED_INTERFACES.items():
        if interface not in interfaces:
            continue
        if all(
            any(key in fixed_ports and requirement.ports.get(key) for key in keys)
            for keys, _ in members
        ) and not any(
            key not in fixed_ports and requirement.ports.get(key)
            for keys, _ in members
            for key in keys
        ):
            continue
        for keys, capability in members:
            # A recipe's programming UART and a separate application UART may
            # coexist. Do not reinterpret fixed device ports as application aliases.
            bindings = {
                key: requirement.ports[key]
                for key in keys
                if key not in fixed_ports and requirement.ports.get(key)
            }
            if len(set(bindings.values())) > 1:
                raise _interface_port_error(requirement, interface, list(keys), conflict=True)
            if not bindings:
                raise _interface_port_error(
                    requirement, interface, [" | ".join(names) for names, _ in members]
                )
            net = next(iter(bindings.values()))
            port = next(iter(bindings))
            requests.append(
                PinAllocationRequest(
                    id=port,
                    net=net,
                    capability=capability,
                    group=interface,
                )
            )
    for capability in ("input", "output", "touch"):
        requests.extend(
            PinAllocationRequest(id=name, net=net, capability=capability)
            for name, net in sorted(requirement.ports.items())
            if name == capability or name.startswith(f"{capability}_")
        )
    for interface, capability in (("pwm", "pwm"), ("adc", "adc")):
        if interface not in interfaces:
            continue
        matching = sorted(
            (name, net)
            for name, net in requirement.ports.items()
            if name == interface or name.startswith(f"{interface}_")
        )
        if not matching:
            raise _interface_port_error(
                requirement, interface, [interface, f"{interface}_<id>"]
            )
        requests.extend(
            PinAllocationRequest(
                id=name,
                net=net,
                capability=capability,
            )
            for name, net in matching
        )
    return requests


def allocate_requirement_pins(
    definition: RecipeDefinition,
    requirement: CircuitRequirement,
    *,
    existing: list[RecipePinAllocation] | None = None,
    required_capabilities: dict[str, str] | None = None,
) -> list[RecipePinAllocation]:
    fixed_ports = {port.name for port in definition.ports}
    requests = [
        request
        for request in requests_from_requirement(requirement, fixed_ports=fixed_ports)
        if request.id not in fixed_ports
    ]
    requested_nets = {request.net for request in requests}
    pins_by_gpio = {
        pin.gpio: pin
        for pin in definition.allocatable_pins
        if pin.gpio is not None and not pin.reserved
    }
    explicit: list[RecipePinAllocation] = []
    for port_name, net in sorted(requirement.ports.items()):
        match = re.fullmatch(r"gpio[_-]?(\d+)", port_name, re.IGNORECASE)
        if match is None or net in requested_nets:
            continue
        pin = pins_by_gpio.get(int(match.group(1)))
        if pin is None:
            continue
        requests.append(
            PinAllocationRequest(
                id=f"explicit_{port_name}",
                net=net,
                capability="gpio",
                allow_strapping=True,
            )
        )
        explicit.append(RecipePinAllocation(net=net, pin=pin.pin, capability="gpio"))
        requested_nets.add(net)
    if required_capabilities:
        requests_by_net = {request.net: request for request in requests}
        fixed_by_net: dict[str, list[str]] = defaultdict(list)
        for allocation in existing or []:
            fixed_by_net[allocation.net].append(allocation.pin)
        for port_name, net in requirement.ports.items():
            match = re.fullmatch(r"gpio[_-]?(\d+)", port_name, re.IGNORECASE)
            if match is not None:
                pin = pins_by_gpio.get(int(match.group(1)))
                fixed_by_net[net].append(pin.pin if pin is not None else "")
        pins_by_number = {pin.pin: pin for pin in definition.allocatable_pins}
        for net, capability in required_capabilities.items():
            request = requests_by_net.get(net)
            if (
                request is None
                or not (
                    request.capability == capability
                    or (
                        capability == "touch"
                        and (
                            request.capability == "input"
                            or (request.capability == "gpio" and request.id.startswith("explicit_"))
                        )
                    )
                )
                or any(
                    prior.net == net
                    and prior.capability != capability
                    and not (capability == "touch" and prior.capability in {"input", "gpio"})
                    for prior in existing or []
                )
            ):
                raise PinAllocationError(
                    "conflicting_pin_capability",
                    f"net {net!r} requires {capability!r} from its typed peer, "
                    "but its explicit MCU function is incompatible",
                    requirement_id=requirement.id,
                    capability=capability,
                )
            constrained = (
                request
                if request.capability == capability
                else request.model_copy(update={"capability": capability})
            )
            fixed = fixed_by_net.get(net, [])
            fixed_pins = set(fixed)
            if (
                len(fixed_pins) > 1
                or any(
                    pin_number not in pins_by_number
                    or not _supports(pins_by_number[pin_number], constrained)
                    for pin_number in fixed
                )
                or any(
                    other_net != net and fixed_pins.intersection(other_pins)
                    for other_net, other_pins in fixed_by_net.items()
                )
            ):
                raise PinAllocationError(
                    "conflicting_pin_capability",
                    f"net {net!r} requires {capability!r}, incompatible with fixed pins {fixed}; "
                    "repair the explicit pin assignment rather than silently reallocating it",
                    requirement_id=requirement.id,
                    capability=capability,
                )
            requests_by_net[net] = constrained
            if fixed:
                explicit.append(RecipePinAllocation(net=net, pin=fixed[0], capability=capability))
        requests = [requests_by_net[request.net] for request in requests]
    allocations = allocate_pins(
        definition.allocatable_pins,
        requests,
        existing=[*(existing or []), *explicit],
        requirement_id=requirement.id,
    )
    if required_capabilities:
        for allocation in allocations:
            if (
                allocation.net in required_capabilities
                and (fixed := fixed_by_net.get(allocation.net))
                and allocation.pin != fixed[0]
            ):
                raise PinAllocationError(
                    "conflicting_pin_capability",
                    f"net {allocation.net!r} cannot retain fixed pin {fixed[0]!r}; "
                    "repair the conflicting pin assignments",
                    requirement_id=requirement.id,
                    capability=required_capabilities[allocation.net],
                )
    return allocations
