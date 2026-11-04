"""Physical values must remain finite instead of becoming null in saved JSON."""

import pytest
from pydantic import ValidationError

from kicraft.design.models import (
    Architecture,
    ArraySpec,
    CircuitRequirement,
    EdgeInterface,
    FormFactor,
    PlacementBoard,
    RecipeSelection,
)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "construct",
    [
        lambda value: FormFactor(shape="circle", size_mm=value),
        lambda value: FormFactor(shape="rounded_rect", corner_radius_mm=value),
        lambda value: FormFactor(shape="chamfered_rect", chamfer_mm=value),
        lambda value: ArraySpec(refs=["D1"], rows=1, cols=1, pitch_mm=value),
        lambda value: ArraySpec(refs=["D1", "D2", "D3"], pattern="ring", radius_mm=value),
        lambda value: ArraySpec(refs=["D1", "D2", "D3"], pattern="ring", start_angle_deg=value),
        lambda value: EdgeInterface(
            name="edge", refs=["J1"], side="left", pitch_mm=value, behavior="castellated",
        ),
        lambda value: PlacementBoard(width_mm=value, height_mm=30, size_search=False),
        lambda value: Architecture(
            sheets=[], power_nets=[], inter_sheet_nets=[], rail_voltages={"VIN": value},
        ),
        lambda value: CircuitRequirement(
            id="reg", sheet="POWER", role="regulator", family="converter",
            parameters={"input_voltage_v": value},
        ),
        lambda value: RecipeSelection(
            recipe="converter@1", instance="reg", sheets={"power": "POWER"},
            parameters={"input_voltage_v": value},
        ),
    ],
    ids=[
        "board-diameter", "corner-radius", "chamfer", "grid-pitch", "ring-radius",
        "ring-angle", "edge-pitch", "fixed-board-width", "rail-voltage",
        "requirement-parameter", "recipe-parameter",
    ],
)
def test_nonfinite_physical_values_are_rejected_before_saving(construct, value):
    with pytest.raises(ValidationError):
        construct(value)
