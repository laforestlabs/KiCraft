
import pytest

from kicraft.design.lowering import lower_requirement
from kicraft.design.models import CircuitRequirement


_AMPLIFIERS = ("MCP6002-I/SN", "LM358DR", "TLV9062IDR")
_SENSORS = ("temperature", "humidity", "ambient-light", "current-sense", "hall-effect")


def _requirement(sensor: str, channels: int, amplifier: str, *, trim=False):
    ports = {"vplus": "AV5", "vminus": "AGND", "analog_out_return": "AGND"}
    for index in range(1, channels + 1):
        ports.update(
            {
                f"sensor_{index}_signal": f"SHUNT_H{index}" if sensor == "current-sense" else f"S{index}",
                f"sensor_{index}_return": "AGND",
                f"conditioned_{index}": f"OUT{index}",
                f"analog_out_{index}": f"OUT{index}",
            }
        )
    if sensor == "current-sense":
        ports.update(
            {
                f"load_return_in_{index}": f"SHUNT_H{index}"
                for index in range(1, channels + 1)
            }
            | {
                f"load_return_out_{index}": "AGND"
                for index in range(1, channels + 1)
            }
        )
    return CircuitRequirement(
        id="conditioner",
        sheet="ANALOG",
        role="analog_block",
        family="analog-conditioner",
        exact_part=amplifier,
        parameters={
            "sensor_type": sensor,
            "channel_count": channels,
            "analog_rail_v": 5.0,
            "output_load_min_ohm": 10_000.0,
            "output_cap_max_pf": 100.0,
            "trim": trim,
        },
        ports=ports,
    )


@pytest.mark.parametrize("amplifier", _AMPLIFIERS)
@pytest.mark.parametrize("sensor", _SENSORS)
@pytest.mark.parametrize("channels", (2, 4, 8))
def test_conditioner_preserves_requested_dual_amplifier_sensor_and_channel_count(amplifier, sensor, channels):
    artifact = lower_requirement(_requirement(sensor, channels, amplifier))
    groups = {group.role: group for group in artifact.groups}
    assert groups["amplifier"].mpn == amplifier
    assert groups["sensor"].quantity == channels
    assert "analog_output" not in groups
    assert {pin.net for pin in artifact.pins if pin.role == "amplifier" and pin.pin in {"1", "7"}} >= {f"OUT{i}" for i in range(1, channels + 1)}


@pytest.mark.parametrize("sensor", _SENSORS)
def test_requested_trim_has_wiper_straps_and_bounded_fixed_network(sensor):
    artifact = lower_requirement(_requirement(sensor, 2, "MCP6002-I/SN", trim=True))
    trims = {(pin.index, pin.pin): pin.net for pin in artifact.pins if pin.role == "trim"}
    assert {(index, "2") for index in range(2)} <= trims.keys()
    assert all(trims[index, "2"] == trims[index, "3"] != trims[index, "1"] for index in range(2))
    groups = {group.role: group for group in artifact.groups}
    assert "gain_feedback" in groups or "divider_bottom" in groups or "emitter_load" in groups


def test_ambient_trim_is_only_the_documented_emitter_rheostat():
    artifact = lower_requirement(_requirement("ambient-light", 2, "MCP6002-I/SN", trim=True))
    by_role = {(pin.role, pin.index, pin.pin): pin.net for pin in artifact.pins}
    assert by_role["divider_bottom", 0, "2"] == "AGND"
    assert by_role["trim", 0, "1"] == by_role["emitter_load", 0, "2"]
    assert by_role["trim", 0, "2"] == by_role["trim", 0, "3"] == "AGND"


def test_current_sense_has_real_low_side_return_and_never_a_sense_placeholder():
    artifact = lower_requirement(_requirement("current-sense", 2, "LM358DR"))
    sensor_pins = {(pin.index, pin.pin): pin.net for pin in artifact.pins if pin.role == "sensor"}
    assert sensor_pins == {(0, "1"): "SHUNT_H1", (0, "2"): "AGND", (1, "1"): "SHUNT_H2", (1, "2"): "AGND"}
    assert all("SENSE" not in str(pin.net) for pin in artifact.pins)


def test_3v3_consumer_selects_tlv9062_documented_reduced_span_without_rewriting_explicit_gain():
    requirement = _requirement("current-sense", 2, "TLV9062IDR")
    artifact = lower_requirement(requirement.model_copy(update={"parameters": {**requirement.parameters, "consumer_input_max_v": 3.3}}))
    groups = {group.role: group for group in artifact.groups}
    assert groups["gain_ground"].value == "4.02k"
    assert groups["gain_feedback"].value == "95.3k"
    assert {calculation.output: calculation.chosen_value for calculation in artifact.calculations}["maximum_output_v"] == "2.471V"
    with pytest.raises(ValueError):
        lower_requirement(requirement.model_copy(update={"parameters": {**requirement.parameters, "consumer_input_max_v": 3.3, "gain": 49.5}}))


def test_trimmed_high_gain_current_observer_requires_actual_buffer_chain():
    from kicraft.eval.analog_artifact_evidence import analog_conditioner_facts

    artifact = lower_requirement(_requirement("current-sense", 2, "TLV9062IDR", trim=True))
    parts, nets = [], {}
    for group in artifact.groups:
        for index in range(group.quantity):
            ref = f"{group.role}_{index}"
            parts.append({"ref": ref, "symbol": group.symbol, "footprint": group.footprint,
                          "value": group.value, "mpn": group.mpn})
    for pin in artifact.pins:
        nets[f"{pin.role}_{pin.index}", pin.pin] = pin.net
    for index in (1, 2):
        ref = f"J{index}"
        parts.append({"ref": ref, "mpn": "TSW-103-07-G-S",
                      "symbol": "Connector_Generic:Conn_01x03",
                      "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"})
        nets[ref, "1"], nets[ref, "2"], nets[ref, "3"] = "AGND", f"OUT{index}", f"SHUNT_H{index}"
    supply = lambda rail, ground: 5.0 if (rail, ground) == ("AV5", "AGND") else None
    facts = analog_conditioner_facts(parts, {}, nets, supply_voltage=supply)
    assert facts["gates"]["external_current_sense_conditioning_function"] == "pass"
    assert set(facts["analog_output_nets"]) == {"OUT1", "OUT2"}
    nets["amplifier_0", "5"] = "OPEN_DIVIDER"
    damaged = analog_conditioner_facts(parts, {}, nets, supply_voltage=supply)
    assert "external_current_sense_conditioning_function" not in damaged["gates"]


@pytest.mark.parametrize("trim", [False, True])
def test_temperature_consumer_limit_covers_full_range_and_unspecified_component_errors(trim):
    requirement = _requirement("temperature", 2, "MCP6002-I/SN", trim=trim)
    for limit in (2.0, 3.3, 4.99):
        with pytest.raises(ValueError):
            lower_requirement(requirement.model_copy(update={
                "parameters": {**requirement.parameters, "consumer_input_max_v": limit},
            }))
    lower_requirement(requirement.model_copy(update={
        "parameters": {**requirement.parameters, "consumer_input_max_v": 5.0},
    }))


def test_eight_isolated_sht30_channels_exceed_switched_capacitance():
    requirement = _requirement("humidity", 8, "MCP6002-I/SN")
    isolated = requirement.model_copy(update={
        "parameters": {**requirement.parameters, "power_cycle_isolation": True},
        "ports": {**requirement.ports, "isolation_vdd": "HOST5",
                  "isolation_enable": "RELEASE",
                  **{f"analog_out_{i}": f"HOST_OUT{i}" for i in range(1, 9)}},
    })
    with pytest.raises(ValueError):
        lower_requirement(isolated)


def test_conditioner_refuses_unsafe_load_or_unbound_output_contract():
    requirement = _requirement("hall-effect", 2, "TLV9062IDR")
    with pytest.raises(ValueError):
        lower_requirement(requirement.model_copy(update={"parameters": {**requirement.parameters, "output_cap_max_pf": 101.0}}))
    broken = requirement.model_copy(update={"ports": {**requirement.ports, "conditioned_1": "A", "analog_out_1": "B"}})
    with pytest.raises(ValueError):
        lower_requirement(broken)


def test_conditioner_leaves_requested_output_connector_to_architecture():
    requirement = _requirement("humidity", 2, "MCP6002-I/SN")
    artifact = lower_requirement(requirement)

    assert "analog_output" not in {group.role for group in artifact.groups}
    with pytest.raises(ValueError):
        lower_requirement(
            requirement.model_copy(
                update={
                    "parameters": {
                        **requirement.parameters,
                        "analog_output_connector": "TSW-103-07-G-S",
                    }
                }
            )
        )


def test_temperature_sensors_have_required_bypass_with_reset_capacitance_bound():
    requirement = _requirement("temperature", 4, "TLV9062IDR")
    isolated = requirement.model_copy(
        update={
            "parameters": {**requirement.parameters, "power_cycle_isolation": True},
            "ports": {
                **requirement.ports,
                "isolation_vdd": "HOST_5V",
                "isolation_enable": "RESET_RELEASE",
                **{
                    f"analog_out_{index}": f"HOST_OUT{index}"
                    for index in range(1, 5)
                },
            },
        }
    )
    artifact = lower_requirement(isolated)
    decoupling = [pin for pin in artifact.pins if pin.role == "sensor_decoupling"]

    assert len(decoupling) == 8
    assert {
        (pin.index, pin.pin, pin.net) for pin in decoupling
    } == {
        *( (index, "1", "AV5") for index in range(4) ),
        *( (index, "2", "AGND") for index in range(4) ),
    }

    eight_channel = _requirement("temperature", 8, "TLV9062IDR").model_copy(
        update={
            "parameters": {
                **requirement.parameters,
                "channel_count": 8,
                "power_cycle_isolation": True,
            },
            "ports": {
                **_requirement("temperature", 8, "TLV9062IDR").ports,
                "isolation_vdd": "HOST_5V",
                "isolation_enable": "RESET_RELEASE",
                **{
                    f"analog_out_{index}": f"HOST_OUT{index}"
                    for index in range(1, 9)
                },
            },
        }
    )
    with pytest.raises(ValueError):
        lower_requirement(eight_channel)


def _temt6000_requirement(*, family="ambient-light-phototransistor"):
    return CircuitRequirement(
        id="light",
        sheet="LIGHT",
        role="sensor",
        family=family,
        exact_part="TEMT6000X01",
        parameters={"analog_rail_v": 3.3, "consumer_input_max_v": 3.3},
        ports={"vplus": "V3P3", "vminus": "AGND", "analog_out": "LIGHT_ADC"},
    )


@pytest.mark.parametrize("family", ("ambient-light-phototransistor", "ambient-light-sensor"))
def test_temt6000_direct_readout_owns_bias_and_filter_with_only_external_ports(family):
    artifact = lower_requirement(_temt6000_requirement(family=family))

    groups = {group.role: group for group in artifact.groups}
    assert groups["sensor"].mpn == "TEMT6000X01"
    assert groups["emitter_load"].value == "10k"
    assert groups["output_filter"].value == "1nF"
    assert {(pin.role, pin.pin, pin.net) for pin in artifact.pins} >= {
        ("sensor", "1", "V3P3"), ("sensor", "2", "LIGHT_ADC"),
        ("emitter_load", "1", "LIGHT_ADC"), ("emitter_load", "2", "AGND"),
        ("output_filter", "1", "LIGHT_ADC"), ("output_filter", "2", "AGND"),
    }


@pytest.mark.parametrize(
    "requirement",
    [
        _temt6000_requirement().model_copy(update={"exact_part": "SFH309"}),
        _temt6000_requirement().model_copy(
            update={"parameters": {"analog_rail_v": 5.0, "consumer_input_max_v": 3.3}}
        ),
        _temt6000_requirement().model_copy(
            update={"ports": {"collector": "V3P3", "emitter": "LIGHT_ADC"}}
        ),
    ],
)
def test_temt6000_direct_readout_refuses_identity_voltage_and_pin_map_shims(requirement):
    with pytest.raises(ValueError):
        lower_requirement(requirement)
