"""Reviewed actuator carrier recipes whose geometry is a board interface."""
from __future__ import annotations

from .models import (
    RecipeComponentGroup as Group,
    RecipeDefinition,
    RecipeElectricalAssertion as Assertion,
    RecipePinSpec as Pin,
    RecipePlacementConstraint as Placement,
    RecipePort as Port,
    RecipeSourceDocument as Source,
)

_A4988_SOCKET = RecipeDefinition(
    recipe="a4988-carrier-socket@1",
    family="a4988-driver-socket",
    exact_part="a4988-carrier-socket",
    maturity="production",
    protected_aliases=("A4988 driver socket", "A4988 carrier socket"),
    identity_aliases=("A4988 carrier socket",),
    required_sheet_roles=("interface",),
    ports=tuple(
        Port(name=name, direction=direction)
        for name, direction in (
            ("vmotor", "power"), ("vdd", "power"), ("gnd", "power"),
            ("enable_n", "input"), ("ms1", "input"), ("ms2", "input"), ("ms3", "input"),
            ("reset_n", "input"), ("sleep_n", "input"), ("step", "input"), ("dir", "input"),
            ("coil_1a", "output"), ("coil_1b", "output"), ("coil_2a", "output"), ("coil_2b", "output"),
        )
    ),
    parts=(
        Group(role="carrier_socket", reference_prefix="J", value="A4988 carrier socket", symbol="a4988-carrier-socket:A4988_CARRIER_SOCKET", footprint="a4988-carrier-socket:A4988_CARRIER_SOCKET_POLolu_1182", sheet_role="interface"),
        Group(role="motor_terminal", reference_prefix="J", value="WJ126V-5.0-4P", symbol="screw-terminal-5mm-4p:WJ126V-5.0-4P", footprint="screw-terminal-5mm-4p:CONN-TH_4P-P5.00_WJ126V-5.0-4P-1", sheet_role="interface", mpn="WJ126V-5.0-04P-14-00A"),
        Group(role="vm_bulk", reference_prefix="C", value="100uF 35V", symbol="Device:C_Polarized", footprint="Capacitor_SMD:C_1210_3225Metric", sheet_role="interface"),
        Group(role="vm_decoupling", reference_prefix="C", value="100nF 50V", symbol="Device:C", footprint="Capacitor_SMD:C_0805_2012Metric", sheet_role="interface"),
        Group(role="enable_pullup", reference_prefix="R", value="100k", symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric", sheet_role="interface"),
        Group(role="control_pulldown", reference_prefix="R", value="100k", symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric", sheet_role="interface", quantity=5),
    ),
    pins=(
        *(Pin(role="carrier_socket", pin=str(index), net=net) for index, net in enumerate(("enable_n", "ms1", "ms2", "ms3", "reset_n", "sleep_n", "step", "dir", "gnd", "vdd", "coil_1b", "coil_1a", "coil_2a", "coil_2b", "gnd", "vmotor"), 1)),
        Pin(role="motor_terminal", pin="1", net="coil_1a"), Pin(role="motor_terminal", pin="2", net="coil_1b"), Pin(role="motor_terminal", pin="3", net="coil_2a"), Pin(role="motor_terminal", pin="4", net="coil_2b"),
        Pin(role="vm_bulk", pin="1", net="vmotor"), Pin(role="vm_bulk", pin="2", net="gnd"), Pin(role="vm_decoupling", pin="1", net="vmotor"), Pin(role="vm_decoupling", pin="2", net="gnd"),
        Pin(role="enable_pullup", pin="1", net="vdd"), Pin(role="enable_pullup", pin="2", net="enable_n"),
        *(Pin(role="control_pulldown", index=index, pin="1", net=net) for index, net in enumerate(("ms1", "ms2", "ms3", "reset_n", "sleep_n"))),
        *(Pin(role="control_pulldown", index=index, pin="2", net="gnd") for index in range(5)),
    ),
    placement_constraints=(Placement(kind="carrier_keepout", role="carrier_socket", parameters={"width_mm": 15.2, "length_mm": 20.3, "row_pitch_mm": 15.24}),),
    electrical_assertions=(Assertion(code="carrier_socket_not_bare_ic", message="This recipe emits Pololu 1182-compatible 2x8 carrier mating rows, not an A4988 WQFN-28 IC footprint. ENABLE is pulled high; RESET and SLEEP are pulled low."),),
    source_documents=(Source(url="https://www.pololu.com/product-info-merged/1182", title="Pololu A4988 stepper motor driver carrier", revision="current", reviewed_date="2026-10-11", sections=("Pinout", "Dimensions", "Power connections")),),
)

WAVE_F_ACTUATOR_RECIPES = (_A4988_SOCKET,)
