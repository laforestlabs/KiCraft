from __future__ import annotations

from copy import deepcopy

import pytest

from kicraft.design.models import BOM, RecipeSelection
from kicraft.design.recipes.data_logging_storage import MICROCHIP_24LC256_I2C_EEPROM
from kicraft.design.recipes.models import ResolvedRecipeSelection
from kicraft.design.recipes.registry import expand_static_definition
from kicraft.design.synthesis.validation import check_net_coverage
from kicraft.design.part_identity import reviewed_part
from kicraft.eval.data_logging_artifact_evidence import data_logging_hardware_facts


def _storage_expansion():
    selection = RecipeSelection(
        recipe=MICROCHIP_24LC256_I2C_EEPROM.recipe,
        instance="logger_store",
        sheets={"storage": "STORAGE"},
        port_bindings={"vdd": "V33", "gnd": "GND", "sda": "I2C_SDA", "scl": "I2C_SCL"},
        requirement_ids=["logger_storage"],
    )
    return expand_static_definition(
        MICROCHIP_24LC256_I2C_EEPROM,
        ResolvedRecipeSelection(selection=selection, parameters={}),
    )


def _part(ref: str, identity: str, **extra):
    record = reviewed_part(identity)
    return {
        "ref": ref,
        "mpn": record.identity,
        "symbol": record.symbol,
        "footprint": record.footprint,
        **extra,
    }


def _logging_fixture():
    parts = [
        _part("U1", "stm32g030f6p6"),
        _part("U2", "24lc256t-i/sn"),
        {"ref": "C1", "value": "100nF"},
        {"ref": "R1", "value": "4.7k"},
        {"ref": "R2", "value": "4.7k"},
    ]
    pads = {
        # STM32G030 fixed I2C contacts, plus four independently ADC-capable pads.
        ("U1", "1"): "I2C_SCL", ("U1", "2"): "I2C_SDA",
        ("U1", "7"): "LIGHT_1", ("U1", "8"): "LIGHT_2",
        ("U1", "9"): "LIGHT_3", ("U1", "10"): "LIGHT_4",
        ("U2", "1"): "GND", ("U2", "2"): "GND", ("U2", "3"): "GND",
        ("U2", "4"): "GND", ("U2", "5"): "I2C_SDA", ("U2", "6"): "I2C_SCL",
        ("U2", "7"): "GND", ("U2", "8"): "V33",
        ("C1", "1"): "V33", ("C1", "2"): "GND",
        ("R1", "1"): "V33", ("R1", "2"): "I2C_SDA",
        ("R2", "1"): "V33", ("R2", "2"): "I2C_SCL",
    }
    return parts, pads


def _facts(parts, pads, *, supply_voltage=None):
    return data_logging_hardware_facts(
        parts,
        {},
        pads,
        mcu_facts={
            "complete_mcu_refs": ["U1"],
            "complete_mcu_supplies": {"U1": {"rail": "V33", "ground": "GND"}},
        },
        analog_facts={"analog_output_nets": ["LIGHT_1", "LIGHT_2", "LIGHT_3", "LIGHT_4"]},
        powered_mcu_refs={"U1"},
        supply_voltage=supply_voltage or (lambda rail, ground: 3.3 if (rail, ground) == ("V33", "GND") else None),
    )


def test_i2c_eeprom_recipe_materializes_writable_default_address_and_support_network():
    expansion = _storage_expansion()
    assert check_net_coverage(BOM(
        parts=expansion.parts,
        connections=expansion.connections,
        no_connect_pins=expansion.no_connect_pins,
    )).ok
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    refs = {part.recipe_role: part.ref for part in expansion.parts}
    assert next(part.footprint for part in expansion.parts if part.ref == refs["storage"]) == (
        "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"
    )
    assert {nets[refs["storage"], pin] for pin in ("1", "2", "3", "7")} == {"GND"}
    assert nets[refs["storage"], "5"] == nets[refs["sda_pullup"], "2"] == "I2C_SDA"
    assert nets[refs["storage"], "6"] == nets[refs["scl_pullup"], "2"] == "I2C_SCL"
    assert nets[refs["decoupling"], "1"] == "V33"


def test_data_logger_hardware_capability_requires_storage_mcu_i2c_and_evidenced_acquisition():
    parts, pads = _logging_fixture()
    facts = _facts(parts, pads)
    assert facts["gates"] == {
        "external_data_logging_function": "pass",
    }
    assert facts["data_logging_storage_addresses"] == ["0x50"]
    assert facts["data_logging_mcu_refs"] == ["U1"]
    one_channel = data_logging_hardware_facts(
        parts, {}, pads, mcu_facts={
            "complete_mcu_refs": ["U1"],
            "complete_mcu_supplies": {"U1": {"rail": "V33", "ground": "GND"}},
        },
        analog_facts={"analog_output_nets": ["LIGHT_1"]}, powered_mcu_refs={"U1"},
        supply_voltage=lambda rail, ground: 3.3 if (rail, ground) == ("V33", "GND") else None,
    )
    assert one_channel["gates"]["external_data_logging_function"] == "pass"
    independent = data_logging_hardware_facts(
        parts, {}, pads, mcu_facts={
            "complete_mcu_refs": ["U1"],
            "complete_mcu_supplies": {"U1": {"rail": "V33", "ground": "GND"}},
        }, analog_facts={},
        independent_input_nets={"LIGHT_1"}, powered_mcu_refs={"U1"},
        supply_voltage=lambda rail, ground: 3.3 if (rail, ground) == ("V33", "GND") else None,
    )
    assert independent["gates"]["external_data_logging_function"] == "pass"

    defects = {
        "disconnected_sda": lambda nets: nets.pop(("U2", "5")),
        "disconnected_scl": lambda nets: nets.pop(("U2", "6")),
        "unpowered_storage": lambda nets: nets.__setitem__(("U2", "8"), "UNPOWERED"),
        "write_protected": lambda nets: nets.__setitem__(("U2", "7"), "V33"),
        "floating_address": lambda nets: nets.__setitem__(("U2", "1"), "FLOAT"),
    }
    for mutate in defects.values():
        defective = deepcopy(pads)
        mutate(defective)
        assert "external_data_logging_function" not in _facts(parts, defective)["gates"]


def test_storage_must_share_mcu_supply_and_return_without_bus_supply_shorts():
    parts, pads = _logging_fixture()
    for original, replacement in (
        ("V33", "V5"),
        ("GND", "ISOLATED_RETURN"),
        ("I2C_SDA", "V33"),
        ("I2C_SCL", "GND"),
    ):
        defective = {pin: replacement if net == original else net for pin, net in pads.items()}
        facts = _facts(
            parts, defective,
            supply_voltage=lambda rail, ground: 5.0 if rail == "V5" else 3.3,
        )
        assert "external_data_logging_function" not in facts["gates"], (original, replacement)


@pytest.mark.parametrize("shared_bus", [False, True])
def test_logged_measurements_construct_writable_storage_on_the_existing_bus(shared_bus):
    from kicraft.design.architecture_intent import derive_architecture
    from kicraft.design.recipes import apply_architecture_recipe_resolution
    from kicraft.server.stage_work_units import probe_architecture_construction

    payload = {
        "sheets": [{"name": "MAIN", "stem": "MAIN", "role": "mcu", "function": "Measurement logger"}],
        "requirements": [
            {"id": "input", "sheet": "MAIN", "role": "power_input", "family": "screw-terminal",
             "parameters": {"rows": 1}, "ties": {"pin2": "GND"}},
            {"id": "mcu", "sheet": "MAIN", "role": "mcu_core", "family": "ch32v003",
             "supply": "V33", "interfaces": ["i2c_controller"], "functional_blocks": ["COLLECT"]},
            {"id": "bus", "sheet": "MAIN", "role": "connector", "family": "qwiic-i2c-connector",
             "ties": {"ground": "GND", "vcc": "V33"}},
        ],
        "signals": [
            {"name": "SDA", "from": "mcu.i2c_sda", "to": "bus.sda"},
            {"name": "SCL", "from": "mcu.i2c_scl", "to": "bus.scl"},
        ],
        "power": {"rails": {"V33": {"voltage": 3.3, "from": "input.pin1"}}},
        "mcu_present": True,
    }
    if shared_bus:
        payload["requirements"].append({
            "id": "peer", "sheet": "MAIN", "role": "connector", "family": "qwiic-i2c-connector",
            "ties": {"ground": "GND", "vcc": "V33"},
        })
        payload["signals"].extend([
            {"name": "SDA", "from": "mcu.i2c_sda", "to": "peer.sda"},
            {"name": "SCL", "from": "mcu.i2c_scl", "to": "peer.scl"},
        ])
    functional = {"blocks": [{
        "name": "COLLECT", "category": "process",
        "purpose": "Collect and manage the ambient-light measurements as logged data.",
    }]}
    architecture = apply_architecture_recipe_resolution(derive_architecture(payload, functional))
    constructed = probe_architecture_construction({"architecture": architecture.model_dump(mode="json")})
    bom = BOM.model_validate(constructed)
    storage = next(part for part in bom.parts if part.mpn == "24LC256T-I/SN")
    pads = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in bom.connections for endpoint in connection.endpoints
    }
    assert tuple(pads[storage.ref, pin] for pin in ("8", "4", "5", "6", "7")) == (
        "V33", "GND", "SDA", "SCL", "GND",
    )


@pytest.mark.parametrize("fault", [None, "sda_open", "scl_open", "unpowered", "address_collision"])
def test_logger_external_adc_requires_a_complete_readable_bus(fault):
    parts, pads = _logging_fixture()
    for pin in ("7", "8", "9", "10"):
        pads.pop(("U1", pin))
    parts.append(_part("U3", "ads1115idgsr"))
    pads.update({
        ("U3", "1"): "GND", ("U3", "3"): "GND", ("U3", "8"): "V33",
        ("U3", "9"): "I2C_SDA", ("U3", "10"): "I2C_SCL",
        **{("U3", str(4 + channel)): f"LIGHT_{1 + channel}" for channel in range(4)},
    })
    if fault == "sda_open":
        pads["U3", "9"] = "OPEN"
    elif fault == "scl_open":
        pads["U3", "10"] = "OPEN"
    elif fault == "unpowered":
        pads["U3", "8"] = "OPEN"
    elif fault == "address_collision":
        parts.append(_part("U4", "ads1115idgsr"))
        pads.update({("U4", pin): net for (ref, pin), net in list(pads.items()) if ref == "U3"})
    facts = _facts(parts, pads)
    assert (facts["gates"].get("external_data_logging_function") == "pass") is (fault is None)
    if fault is None:
        assert facts["channel_counts"]["data_logging_acquisition"] == 4
