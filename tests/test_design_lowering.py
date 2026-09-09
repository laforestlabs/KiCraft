import math

import pytest

from kicraft.design.lowering import lower_requirement, registered_lowerers
from kicraft.design.models import CircuitRequirement
from kicraft.server.stage_work_units import (
    StageWorkUnit,
    deterministic_bom_candidate,
    deterministic_wiring_candidate,
    merge_bom_units,
    plan_stage_work_units,
    validate_unit_candidate,
)


def _requirement(family: str, *, parameters=None, ports=None) -> CircuitRequirement:
    return CircuitRequirement(
        id="block",
        sheet="MAIN",
        role="analog_block",
        family=family,
        parameters=parameters or {},
        ports=ports or {},
    )


def _state(requirement: CircuitRequirement, parts=None) -> dict:
    return {
        "architecture": {
            "sheets": [{"name": "MAIN"}],
            "requirements": [requirement.model_dump(mode="json")],
            "recipe_selections": [],
            "power_nets": ["GND"],
            "inter_sheet_nets": [],
        },
        "bom": {"parts": parts or []},
    }


def _pinout(count: int) -> dict:
    return {"pins": [{"number": str(index)} for index in range(1, count + 1)]}


def test_lowerer_registry_matches_exact_family_without_ordered_fallback():
    assert lower_requirement(_requirement("pin_header", ports={"one": "N"})) is None
    assert lower_requirement(_requirement("unknown", ports={"one": "N"})) is None
    ids = [lowerer.lowerer_id for lowerer in registered_lowerers()]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_pin_header_lowerer_emits_bom_and_exact_pin_ownership():
    requirement = _requirement(
        "pin-header",
        parameters={"rows": 2},
        ports={"one": "D0", "two": "D1", "three": "3V3", "four": "GND"},
    )
    artifact = lower_requirement(requirement)
    assert artifact is not None
    assert artifact.lowerer_id == "pin-header@1"
    assert artifact.groups[0].symbol == "Connector_Generic:Conn_02x02"
    assert [(pin.pin, pin.net) for pin in artifact.pins] == [
        ("1", "D0"),
        ("2", "D1"),
        ("3", "3V3"),
        ("4", "GND"),
    ]


@pytest.mark.parametrize(
    ("family", "parameters", "ports"),
    [
        (
            "led-current-resistor",
            {"rail_voltage": 3.3, "led_vf": 2.0, "target_current_ma": 5},
            {"drive": "STATUS", "gnd": "GND"},
        ),
        (
            "voltage-divider",
            {"input_voltage": 12, "target_voltage": 3.3, "bottom_resistance_ohm": 10000},
            {"input": "VIN", "output": "SENSE", "gnd": "GND"},
        ),
        (
            "rc-lowpass",
            {"cutoff_hz": 1000, "resistance_ohm": 10000},
            {"input": "RAW", "output": "FILTERED", "gnd": "GND"},
        ),
    ],
)
def test_calculated_lowerers_choose_standard_values_with_bounded_error(family, parameters, ports):
    artifact = lower_requirement(_requirement(family, parameters=parameters, ports=ports))
    assert artifact is not None
    assert artifact.calculations
    assert "unrounded" not in artifact.calculations[0].tolerance


def test_voltage_divider_rejects_unachievable_error_bound():
    requirement = _requirement(
        "voltage-divider",
        parameters={
            "input_voltage": 12,
            "target_voltage": math.pi,
            "bottom_resistance_ohm": 10000,
            "max_error_percent": 0.001,
        },
        ports={"input": "VIN", "output": "SENSE", "gnd": "GND"},
    )
    assert lower_requirement(requirement) is None


@pytest.mark.parametrize("capacitance", [10, 400])
def test_i2c_pullup_lowerer_accepts_bus_capacitance_boundaries(capacitance):
    requirement = _requirement(
        "i2c-pullups",
        parameters={"speed_hz": 400000, "bus_capacitance_pf": capacitance, "voltage": 3.3},
        ports={"vdd": "3V3", "sda": "SDA", "scl": "SCL"},
    )
    assert lower_requirement(requirement) is not None


@pytest.mark.parametrize("capacitance", [9, 401])
def test_i2c_pullup_lowerer_rejects_out_of_range_capacitance(capacitance):
    requirement = _requirement(
        "i2c-pullups",
        parameters={"speed_hz": 400000, "bus_capacitance_pf": capacitance, "voltage": 3.3},
        ports={"vdd": "3V3", "sda": "SDA", "scl": "SCL"},
    )
    assert lower_requirement(requirement) is None


def test_r2r_bom_provenance_drives_wiring_from_the_same_artifact():
    requirement = _requirement(
        "r2r-ladder",
        parameters={"bits": 3, "r_value": "10k", "two_r_value": "20k"},
        ports={"bit0": "D0", "bit1": "D1", "bit2": "D2", "output": "DAC", "gnd": "GND"},
    )
    state = _state(requirement)
    bom_unit = StageWorkUnit("bom-r000", "bom", "MAIN", requirement_ids=(requirement.id,))
    candidate = validate_unit_candidate(bom_unit, {"groups": [], "arrays": []}, state, {})
    merged, ref_to_unit, ref_to_lowering = merge_bom_units(
        (bom_unit,), {bom_unit.unit_id: candidate}, state
    )
    assert ref_to_unit == {f"R{index}": "bom-r000" for index in range(1, 7)}
    assert ref_to_lowering["R1"] == {
        "resolution_id": "r2r-ladder@1",
        "lowering_requirement_id": "block",
        "lowering_role": "series",
        "lowering_index": 0,
    }

    parts = []
    next_ref = 1
    for group in merged["groups"]:
        for _ in range(group["quantity"]):
            ref = f"R{next_ref}"
            next_ref += 1
            parts.append(
                {
                    "ref": ref,
                    "sheet": "MAIN",
                    "symbol": group["symbol"],
                    "value": group["value"],
                    "resolution_source": "lowerer",
                    **ref_to_lowering[ref],
                }
            )
    state["bom"] = {"parts": parts}
    extras = {"symbol_pinouts": {part["ref"]: _pinout(2) for part in parts}}
    units = plan_stage_work_units("wiring", state, extras)
    assert len(units) == 1
    assert units[0].planned_resolution_source == "lowerer"
    assert units[0].lowerer_ids == ("r2r-ladder@1",)

    wiring = deterministic_wiring_candidate(units[0], state, extras)
    assert wiring is not None
    by_pin = {(row["ref"], row["pin"]): row["net"] for row in wiring["pins"]}
    assert by_pin[("R1", "1")] == "DAC"
    assert by_pin[("R3", "1")] == "D2"
    assert by_pin[("R6", "2")] == "GND"


def test_bom_candidate_requires_typed_requirement_ownership():
    requirement = _requirement("pin-header", ports={"one": "GND"})
    state = _state(requirement)
    assert deterministic_bom_candidate(StageWorkUnit("bom-s000", "bom", "MAIN"), state) is None


def test_lowerer_groups_use_loadable_symbols_and_footprints():
    from pathlib import Path

    from kicraft.design.synthesis.footprint_library import lookup_footprint
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    requirements = [
        _requirement("pin-header", parameters={"rows": 1}, ports={"p0": "GND", "p1": "3V3"}),
        _requirement("screw-terminal", ports={"p0": "VIN", "p1": "GND"}),
        _requirement(
            "fpc-header-breakout",
            parameters={"pitch_mm": 0.5},
            ports={f"p{index}": f"N{index}" for index in range(10)},
        ),
        _requirement("test-points", ports={"a": "A", "b": "B"}),
        _requirement(
            "connector-bank",
            parameters={"channels": 2},
            ports={"vdd": "5V", "gnd": "GND", "signal0": "S0", "signal1": "S1"},
        ),
        _requirement(
            "r2r-ladder",
            parameters={"bits": 2, "r_value": "10k", "two_r_value": "20k"},
            ports={"bit0": "D0", "bit1": "D1", "output": "DAC", "gnd": "GND"},
        ),
        _requirement(
            "led-current-resistor",
            parameters={"rail_voltage": 3.3, "led_vf": 2.0, "target_current_ma": 5},
            ports={"drive": "LED", "gnd": "GND"},
        ),
        _requirement(
            "voltage-divider",
            parameters={"input_voltage": 12, "target_voltage": 3.3, "bottom_resistance_ohm": 10000},
            ports={"input": "VIN", "output": "SENSE", "gnd": "GND"},
        ),
        _requirement(
            "rc-lowpass",
            parameters={"cutoff_hz": 1000, "resistance_ohm": 10000},
            ports={"input": "RAW", "output": "FILTERED", "gnd": "GND"},
        ),
        _requirement(
            "i2c-pullups",
            parameters={"speed_hz": 400000, "bus_capacitance_pf": 100, "voltage": 3.3},
            ports={"vdd": "3V3", "sda": "SDA", "scl": "SCL"},
        ),
        _requirement(
            "open-drain-pullup",
            parameters={"resistance": "10k"},
            ports={"vdd": "3V3", "signal": "IRQ"},
        ),
        _requirement(
            "switch-input",
            parameters={"pull_policy": "external", "resistance": "10k"},
            ports={"signal": "BUTTON", "gnd": "GND", "vdd": "3V3"},
        ),
        _requirement(
            "explicit-decoupling",
            parameters={"count": 2, "value": "100nF"},
            ports={"vdd": "3V3", "gnd": "GND"},
        ),
    ]
    for requirement in requirements:
        artifact = lower_requirement(requirement)
        assert artifact is not None, requirement.family
        for group in artifact.groups:
            lookup_pins(group.symbol, project_root=Path("."))
            lookup_footprint(group.footprint, project_root=Path("."))
