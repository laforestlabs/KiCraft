from collections import Counter

import pytest

from kicraft.design.models import CircuitRequirement, RecipeSelection
from kicraft.design.recipes import expand_recipe
from kicraft.design.recipes.models import RecipeAllocatablePin, RecipeDefinition
from kicraft.design.recipes.pin_allocator import (
    PinAllocationError,
    PinAllocationRequest,
    allocate_pins,
    allocate_requirement_pins,
)
from kicraft.design.recipes.registry import get_recipe
from kicraft.design.recipes.rp2040_minimal_v2 import effective_rp2040_definition


@pytest.mark.parametrize(("output_count", "touch_count"), [(6, 2), (5, 6)])
def test_attiny1614_touch_and_leds_expand_to_disjoint_physical_pins(output_count, touch_count):
    # DS40002204A Table 5-1: the SOIC14 has exactly six PTC Y-lines.
    # Five outputs plus all six sensors is feasible, but broad-GPIO-first is not.
    ptc_pins = {"2", "3", "4", "5", "8", "9"}
    application_ports = {
        **{f"output_{index}": f"LED_{index}" for index in range(output_count)},
        **{f"touch_{index}": f"TOUCH_{index}" for index in range(touch_count)},
    }
    requirement = CircuitRequirement(
        id="controller",
        sheet="MCU",
        role="mcu_core",
        family="attiny1614",
        ports={"vdd": "VBAT", "gnd": "GND", "updi": "UPDI", **application_ports},
    )
    definition = get_recipe("attiny1614-updi-minimal@1")
    allocations = allocate_requirement_pins(definition, requirement)
    assert Counter(allocation.net for allocation in allocations) == Counter(
        application_ports.values()
    )
    assert len({allocation.pin for allocation in allocations}) == output_count + touch_count
    assert {allocation.pin for allocation in allocations} <= {
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "11",
        "12",
        "13",
    }
    assert {
        allocation.pin for allocation in allocations if allocation.capability == "touch"
    } <= ptc_pins
    assert allocate_requirement_pins(definition, requirement, existing=allocations) == allocations

    expansion = expand_recipe(
        RecipeSelection(
            recipe=definition.recipe,
            instance="main",
            sheets={"mcu": "MCU"},
            port_bindings={"vdd": "VBAT", "gnd": "GND", "updi": "UPDI"},
            requirement_ids=[requirement.id],
            pin_allocations=allocations,
        )
    )
    mcu = next(part for part in expansion.parts if part.mpn == "ATTINY1614-SSNR")
    mcu_connections = [
        (connection.net_name, endpoint.pin)
        for connection in expansion.connections
        for endpoint in connection.endpoints
        if endpoint.ref == mcu.ref
    ]
    expected = Counter((allocation.net, allocation.pin) for allocation in allocations)
    assert Counter(mcu_connections) == expected + Counter(
        {("VBAT", "1"): 1, ("GND", "14"): 1, ("UPDI", "10"): 1}
    )
    assert (
        Counter(
            (pin.net, pin.pin)
            for pin in expansion.ownership.pins
            if pin.owner == "allocator" and pin.ref == mcu.ref and pin.net is not None
        )
        == expected
    )
    assert {pin.pin for pin in expansion.no_connect_pins if pin.ref == mcu.ref}.isdisjoint(
        allocation.pin for allocation in allocations
    )


@pytest.mark.parametrize("family", ["attiny402", "attiny412"])
def test_unreviewed_tinyavr_packages_do_not_inherit_touch_capability(family):
    requirement = CircuitRequirement(
        id="controller",
        sheet="MCU",
        role="mcu_core",
        family=family,
        ports={"touch_0": "SENSOR"},
    )
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability") as rejected:
        allocate_requirement_pins(get_recipe(f"{family}-updi-minimal@1"), requirement)
    assert rejected.value.capability == "touch"


def test_attiny1614_seventh_touch_channel_cannot_use_non_ptc_gpio_or_updi():
    requirement = CircuitRequirement(
        id="controller",
        sheet="MCU",
        role="mcu_core",
        family="attiny1614",
        ports={f"touch_{index}": f"SENSOR_{index}" for index in range(7)},
    )
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability") as rejected:
        allocate_requirement_pins(get_recipe("attiny1614-updi-minimal@1"), requirement)
    assert rejected.value.capability == "touch"


def test_scarce_touch_allocation_preserves_explicit_gpio_binding():
    definition = RecipeDefinition(
        recipe="test-touch@1",
        required_sheet_roles=("mcu",),
        parts=(),
        pins=(),
        allocatable_pins=(
            RecipeAllocatablePin(role="mcu", pin="1", gpio=0, capabilities=("gpio", "touch")),
            RecipeAllocatablePin(role="mcu", pin="2", gpio=1, capabilities=("gpio", "touch")),
            RecipeAllocatablePin(role="mcu", pin="3", gpio=2, capabilities=("gpio",)),
        ),
    )
    requirement = CircuitRequirement(
        id="controller",
        sheet="MCU",
        role="mcu_core",
        family="test-touch",
        ports={"gpio0": "FIXED", "output_0": "LED", "touch_0": "SENSOR"},
    )
    allocations = allocate_requirement_pins(definition, requirement)
    assert {allocation.net: allocation.pin for allocation in allocations} == {
        "FIXED": "1",
        "LED": "3",
        "SENSOR": "2",
    }


def test_touch_scarcity_order_preserves_only_legal_contiguous_bus_window():
    pins = (
        RecipeAllocatablePin(role="mcu", pin="1", gpio=0, capabilities=("gpio",)),
        RecipeAllocatablePin(role="mcu", pin="2", gpio=1, capabilities=("gpio", "touch")),
        RecipeAllocatablePin(role="mcu", pin="3", gpio=4, capabilities=("gpio", "touch")),
    )
    requests = [
        PinAllocationRequest(id="a_touch", net="SENSOR", capability="touch"),
        *(
            PinAllocationRequest(
                id=f"data_{index}",
                net=f"DATA_{index}",
                capability="output",
                group="z_bus",
                contiguous=True,
            )
            for index in range(2)
        ),
    ]
    allocations = allocate_pins(pins, requests)
    assert {allocation.net: allocation.pin for allocation in allocations} == {
        "DATA_0": "1",
        "DATA_1": "2",
        "SENSOR": "3",
    }
    assert allocate_pins(pins, list(reversed(requests))) == allocations
    assert allocate_pins(pins, requests, existing=allocations) == allocations


def _rp2040_requirement(*, adc_count=4, **ports):
    return CircuitRequirement(
        id="controller", sheet="MCU", role="mcu_core", family="rp2040",
        ports={**{f"adc_{i}": f"SENSOR_{i}" for i in range(adc_count)}, **ports},
    )


def _rp2040_definition(*, exposed=30, bindings=None):
    definition = get_recipe("rp2040-minimal@4")
    return effective_rp2040_definition(
        definition, {**definition.parameter_defaults, "gpio_exposed": exposed}, bindings or {}
    )


def test_rp2040_four_adc_allocations_reach_physical_mcu_and_exposed_contacts():
    allocations = allocate_requirement_pins(_rp2040_definition(), _rp2040_requirement())
    assert {row.pin for row in allocations} == {"38", "39", "40", "41"}
    assert {row.capability for row in allocations} == {"adc"}
    assert len({row.net for row in allocations}) == 4
    expansion = expand_recipe(RecipeSelection(
        recipe="rp2040-minimal@4", instance="main",
        sheets={"mcu": "MCU", "io": "IO"},
        port_bindings={"vdd": "+3V3", "gnd": "GND", "usb_dm": "DM", "usb_dp": "DP"},
        pin_allocations=allocations,
    ))
    mcu = next(part for part in expansion.parts if part.recipe_role == "mcu")
    contacts = {part.ref for part in expansion.parts if part.recipe_role == "cast_right"}
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections for endpoint in connection.endpoints
    }
    for allocation in allocations:
        assert nets[mcu.ref, allocation.pin] == allocation.net
        assert sum(nets.get((ref, "1")) == allocation.net for ref in contacts) == 1
    assert {pin.pin for pin in expansion.no_connect_pins if pin.ref == mcu.ref}.isdisjoint(
        row.pin for row in allocations
    )


def test_rp2040_adc_capacity_respects_exposed_gpio_limit():
    with pytest.raises(PinAllocationError):
        allocate_requirement_pins(_rp2040_definition(), _rp2040_requirement(adc_count=5))
    definition = _rp2040_definition(exposed=29)
    allocations = allocate_requirement_pins(definition, _rp2040_requirement(adc_count=3))
    assert {row.pin for row in allocations} == {"38", "39", "40"}
    with pytest.raises(PinAllocationError):
        allocate_requirement_pins(definition, _rp2040_requirement())
    expansion = expand_recipe(RecipeSelection(
        recipe="rp2040-minimal@4", instance="limited",
        sheets={"mcu": "MCU", "io": "IO"},
        parameters={"gpio_exposed": 29},
        port_bindings={"vdd": "+3V3", "gnd": "GND", "usb_dm": "DM", "usb_dp": "DP"},
        pin_allocations=allocations,
    ))
    mcu = next(part for part in expansion.parts if part.recipe_role == "mcu")
    assert "41" in {pin.pin for pin in expansion.no_connect_pins if pin.ref == mcu.ref}
    assert all(
        endpoint.pin != "41"
        for connection in expansion.connections for endpoint in connection.endpoints
        if endpoint.ref == mcu.ref
    )


def test_rp2040_adc_cannot_share_explicit_gpio_with_another_net():
    requirement = _rp2040_requirement(gpio26="DIGITAL")
    with pytest.raises(PinAllocationError):
        allocate_requirement_pins(_rp2040_definition(), requirement)


@pytest.mark.parametrize("gpio", [0, 1])
def test_rp2040_uart_reservation_rejects_explicit_gpio_collision(gpio):
    ports = {"uart_tx": "TX", "uart_rx": "RX", f"gpio{gpio}": "OTHER"}
    with pytest.raises((PinAllocationError, ValueError)):
        definition = _rp2040_definition(bindings=ports)
        allocate_requirement_pins(definition, _rp2040_requirement(adc_count=0, **ports))


def test_rp2040_uart_reservation_leaves_adc_pins_available():
    ports = {"uart_tx": "TX", "uart_rx": "RX"}
    allocations = allocate_requirement_pins(
        _rp2040_definition(bindings=ports), _rp2040_requirement(**ports)
    )
    by_net = {row.net: row.pin for row in allocations}
    assert by_net["TX"] == "2"
    assert by_net["RX"] == "3"
    assert {pin for net, pin in by_net.items() if net.startswith("SENSOR_")} == {
        "38", "39", "40", "41",
    }
