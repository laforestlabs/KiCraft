"""Stable capability-based allocation for recipe-owned MCU application pins."""

from __future__ import annotations

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
    ):
        self.code = code
        self.requirement_id = requirement_id
        self.capability = capability
        self.count = count
        super().__init__(f"{code}: {message}")


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
    }:
        return False
    return wanted in available or (wanted in {"input", "output"} and "gpio" in available)


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
    existing_map = _existing_by_net(existing)
    pin_by_number = {pin.pin: pin for pin in pins}
    used: set[str] = set()
    result: list[RecipePinAllocation] = []
    groups: dict[str, list[PinAllocationRequest]] = defaultdict(list)
    for request in requests:
        groups[request.group or request.id].append(request)

    for group_name in sorted(groups):
        group = sorted(groups[group_name], key=_request_key)
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
                or not _supports(pin, request)
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
                gpios = [
                    pin_by_number[allocation.pin].gpio for allocation in retained
                ]
                concrete_gpios = sorted(
                    gpio for gpio in gpios if gpio is not None
                )
                if len(concrete_gpios) != len(gpios) or concrete_gpios != list(
                    range(concrete_gpios[0], concrete_gpios[0] + len(gpios))
                ):
                    retained = []
            if retained:
                used.update(retained_pins)
                result.extend(retained)
                continue

        candidates = [
            pin for pin in sorted(pins, key=_pin_key) if pin.pin not in used
        ]
        chosen: list[RecipeAllocatablePin] = []
        if any(request.contiguous for request in group):
            by_gpio = {
                pin.gpio: pin for pin in candidates if pin.gpio is not None
            }
            for start in sorted(by_gpio):
                window = [
                    by_gpio.get(start + offset) for offset in range(len(group))
                ]
                if any(pin is None for pin in window):
                    continue
                concrete = [pin for pin in window if pin is not None]
                if all(
                    _supports(pin, request)
                    for pin, request in zip(concrete, group, strict=True)
                ):
                    chosen = concrete
                    break
        else:
            remaining = list(candidates)
            for request in group:
                match = next(
                    (pin for pin in remaining if _supports(pin, request)),
                    None,
                )
                if match is None:
                    chosen = []
                    break
                chosen.append(match)
                remaining.remove(match)
        if len(chosen) != len(group):
            missing = (
                group[len(chosen)].capability
                if len(chosen) < len(group)
                else group[0].capability
            )
            raise PinAllocationError(
                "unsatisfied_pin_capability",
                f"group {group_name!r} needs {len(group)} legal pins; "
                f"missing {missing!r}",
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
                raise PinAllocationError(
                    "missing_interface_port",
                    f"parallel_output requires port {port!r}",
                    requirement_id=requirement.id,
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
    fixed_interfaces = {
        "i2c_controller": (("sda", "i2c-sda"), ("scl", "i2c-scl")),
        "spi_controller": (
            ("sclk", "spi-sclk"),
            ("mosi", "spi-mosi"),
            ("miso", "spi-miso"),
            ("cs", "spi-cs"),
        ),
        "uart": (("tx", "uart-tx"), ("rx", "uart-rx")),
    }
    for interface, members in fixed_interfaces.items():
        if interface not in interfaces:
            continue
        for port, capability in members:
            net = requirement.ports.get(port)
            if not net:
                raise PinAllocationError(
                    "missing_interface_port",
                    f"{interface} requires port {port!r}",
                    requirement_id=requirement.id,
                )
            requests.append(
                PinAllocationRequest(
                    id=f"{interface}_{port}",
                    net=net,
                    capability=capability,
                    group=interface,
                )
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
            raise PinAllocationError(
                "missing_interface_port",
                f"{interface} requires at least one bound port",
                requirement_id=requirement.id,
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
) -> list[RecipePinAllocation]:
    requests = requests_from_requirement(requirement)
    return allocate_pins(
        definition.allocatable_pins,
        requests,
        existing=existing,
        requirement_id=requirement.id,
    )
