"""Reviewed linear Hall acquisition circuits."""
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


AH49ENTR_G1_ANALOG_HALL = RecipeDefinition(
    recipe="ah49entr-g1-analog-hall@1",
    family="hall-effect",
    exact_part="AH49ENTR-G1",
    # Do not make untyped Hall switches silently resolve to an analog sensor.
    default_for_family=False,
    maturity="production",
    protected_aliases=("AH49ENTR-G1",),
    identity_aliases=("AH49ENTR-G1",),
    required_sheet_roles=("sensor",),
    parameter_defaults={"supply_voltage": 3.3},
    allowed_parameters={"supply_voltage": (3.3,)},
    ports=(Port(name="vdd", direction="power"), Port(name="gnd", direction="power"), Port(name="out", direction="output")),
    parts=(
        Group(role="sensor", reference_prefix="U", value="AH49ENTR-G1", symbol="diodes-ah49entr-g1:AH49ENTR-G1", footprint="diodes-ah49entr-g1:SOT-23-3_Diodes_AH49E", sheet_role="sensor", mpn="AH49ENTR-G1"),
        Group(role="decoupling", reference_prefix="C", value="100nF", symbol="Device:C", footprint="Capacitor_SMD:C_0603_1608Metric", sheet_role="sensor"),
    ),
    pins=(
        Pin(role="sensor", pin="1", net="vdd"),
        Pin(role="sensor", pin="2", net="out"),
        Pin(role="sensor", pin="3", net="gnd"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
    ),
    placement_constraints=(Placement(kind="owned_bypass", role="decoupling", parameters={"anchor_role": "sensor", "power_net": "vdd", "ground_net": "gnd", "max_mm": 3.0}),),
    electrical_assertions=(Assertion(code="ah49e_analog_output", message="AH49E SOT-23-3 pin 1 is VCC, pin 2 is the ratiometric linear output, and pin 3 is GND. The 3.3 V supply and local bypass constrain the native circuit; magnetic performance and external cabling remain unmeasured."),),
    source_documents=(Source(url="https://www.diodes.com/datasheet/download/AH49E.pdf", title="Diodes AH49E linear Hall-effect IC", revision="DS36517 Rev. 2-2, April 2025", reviewed_date="2026-11-04", sections=("Pin Descriptions", "Recommended Operating Conditions", "Electrical Characteristics", "Ordering Information", "SOT-23-3 Package Outline")),),
)

def expand_ah49entr_g1_analog_hall(resolved):
    """Never allow a saved selection to short the sensor's three ports."""
    from .registry import expand_static_definition

    ports = resolved.selection.port_bindings
    if len({ports["vdd"], ports["gnd"], ports["out"]}) != 3:
        raise ValueError("AH49ENTR-G1 vdd, gnd and out must be distinct nets")
    return expand_static_definition(AH49ENTR_G1_ANALOG_HALL, resolved)


WAVE_F_SENSING_RECIPES = (AH49ENTR_G1_ANALOG_HALL,)
