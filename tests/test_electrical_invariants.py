"""Focused regressions for GAP1 device, transfer, and typed-value invariants."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from kicraft.design.synthesis import validation


def _part(ref: str, value: str, *, mpn: str | None = None, sheet: str = "POWER"):
    return SimpleNamespace(ref=ref, value=value, mpn=mpn, symbol="Test:Part", sheet=sheet)


def _bom(parts, nets):
    return SimpleNamespace(
        parts=parts,
        connections=[
            SimpleNamespace(
                net_name=net,
                endpoints=[SimpleNamespace(ref=ref, pin=pin) for ref, pin in endpoints],
            )
            for net, endpoints in nets.items()
        ],
    )


_FACTS = (
    {
        "mpn": "TPS54331DDAR",
        "pins": {"boot": "1", "vin": "2", "ph": "8"},
        "bootstrap_capacitance_f": 100e-9,
        "vin_min_v": 3.5,
        "vin_max_v": 28.0,
        "power_transfer": {"from_pin": "2", "to_pin": "8"},
    },
    {
        "mpn": "TPS5430",
        "pins": {"vin": "VIN"},
        "vin_min_v": 5.5,
        "vin_max_v": 36.0,
    },
    {
        "mpn": "AL8860",
        "port_pins": {"set": "SET", "vin": "VIN", "switch": "SW", "ground": "GND"},
        "power_transfer": {"from_pin": "SW", "to_pin": "GND"},
        "operating_limits": {"continuous_output_a": 1.5},
        "current_feedback": {
            "sense_pin": "SET",
            "reference_pin": "VIN",
            "sense_voltage_v": 0.1,
            "sense_tolerance": 0.04,
            "topology": "high_side_sense_low_side_switch",
        },
        "support_network": {
            "input_decoupling": {
                "positive_pin": "VIN",
                "negative_pin": "GND",
                "capacitance_min_uf": 10,
            },
            "catch_diode": {"anode_pin": "SW", "cathode_pin": "VIN"},
        },
    },
    {
        "mpn": "WRA2412S-3WR2",
        "port_pins": {
            "input_positive": "VIN",
            "input_return": "GND",
            "output_positive": "+VO",
            "output_negative": "-VO",
            "output_common": "0V",
        },
        "operating_limits": {"input_min_v": 18.0, "input_max_v": 36.0},
        "power_transfer": {
            "isolated": True,
            "paths": [{"from_pin": "VIN", "to_pin": "+VO"}, {"from_pin": "VIN", "to_pin": "-VO"}],
        },
    },
    {
        # The real DRV8833 record, in miniature: its supply rating lives under the motor-supply
        # domain and its port is `vm`, so an input-spelling-only reader sees no voltage input.
        "mpn": "DRV8833PWPR",
        "port_pins": {"vm": "12", "ain1": "16", "aout1": "2", "ground": "13"},
        "operating_limits": {
            "motor_supply_min_v": 2.7,
            "motor_supply_max_v": 10.8,
            "continuous_current_per_channel_a": 1.5,
        },
    },
)


@pytest.fixture
def reviewed(monkeypatch):
    monkeypatch.setattr(
        validation,
        "_reviewed_fact_for_part",
        lambda part: next(
            (
                fact
                for fact in _FACTS
                if str(part.mpn or part.value).casefold()
                == str(fact.get("mpn") or "").casefold()
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        validation,
        "_pin_info_by_ref",
        lambda _bom: (
            {
                "U1": {
                    "1": {"name": "BOOT", "type": "passive"},
                    "2": {"name": "VIN", "type": "power_in"},
                    "8": {"name": "PH", "type": "power_out"},
                },
                "U2": {
                    "1": {"name": "VIN", "type": "power_in"},
                },
                "U3": {
                    "1": {"name": "SET", "type": "input"},
                    "2": {"name": "GND", "type": "power_in"},
                    "3": {"name": "GND", "type": "power_in"},
                    "4": {"name": "CTRL", "type": "input"},
                    "5": {"name": "SW", "type": "power_out"},
                    "6": {"name": "SW", "type": "power_out"},
                    "7": {"name": "NC", "type": "passive"},
                    "8": {"name": "VIN", "type": "power_in"},
                    "9": {"name": "EP", "type": "power_in"},
                },
                "D3": {
                    "1": {"name": "A", "type": "passive"},
                    "2": {"name": "K", "type": "passive"},
                },
                "D4": {
                    "1": {"name": "A", "type": "passive"},
                    "2": {"name": "K", "type": "passive"},
                },
                "U4": {
                    "12": {"name": "VM", "type": "power_in"},
                    "13": {"name": "GND", "type": "power_in"},
                },
            },
            {},
        ),
    )


def test_tps54331_bootstrap_requires_direct_100nf_between_boot_and_ph(reviewed):
    bom = _bom(
        [_part("U1", "TPS54331DDAR", mpn="TPS54331DDAR"), _part("C3", "100nF")],
        {
            "VIN": [("U1", "2")],
            "BOOT": [("U1", "1"), ("C3", "1")],
            "PH": [("U1", "8"), ("C3", "2")],
        },
    )
    assert validation.check_reviewed_device_support_networks(bom).ok

    bom.parts[1].value = "10nF"
    result = validation.check_reviewed_device_support_networks(bom)
    assert not result.ok
    assert "E_BOOTSTRAP_SUPPORT" in result.offenders[0]


def test_tps5430_rejects_typed_five_volt_input(reviewed):
    bom = _bom([_part("U2", "TPS5430", mpn="TPS5430")], {"VBUS": [("U2", "1")]})
    architecture = SimpleNamespace(rail_voltages={"VBUS": 5.0})
    result = validation.check_reviewed_input_operating_ranges(architecture, bom)
    assert not result.ok
    assert "E_INPUT_OPERATING_RANGE" in result.offenders[0]
    assert "5.5–36V" in result.offenders[0]


def _conversion_requirement(*, output="VOUT", output_domain="GND"):
    return SimpleNamespace(
        id="buck",
        family="buck-converter",
        role="regulator",
        sheet="POWER",
        obligations=[SimpleNamespace(kind="conversion")],
        ports={"input": "VIN", "output": output, "gnd": "GND"},
        declared_interface=SimpleNamespace(
            ports=[
                SimpleNamespace(key="input", reference_domain="GND"),
                SimpleNamespace(key="output", reference_domain=output_domain),
            ]
        ),
    )


def test_reviewed_buck_transfer_needs_converter_and_inductor_not_capacitor(reviewed):
    requirement = _conversion_requirement()
    architecture = SimpleNamespace(requirements=[requirement])
    bom = _bom(
        [_part("U1", "TPS54331DDAR", mpn="TPS54331DDAR"), _part("L1", "15uH")],
        {"VIN": [("U1", "2")], "SW": [("U1", "8"), ("L1", "1")], "VOUT": [("L1", "2")]},
    )
    assert validation.check_reviewed_power_transfer(architecture, bom).ok

    bom.parts[1] = _part("C1", "22uF")
    result = validation.check_reviewed_power_transfer(architecture, bom)
    assert not result.ok
    assert "E_POWER_TRANSFER" in result.offenders[0]


def test_every_reviewed_buck_proves_its_own_source_to_load_path():
    """§9.39's graph must reach the load for every reviewed buck, from the record's own symbol.

    The frozen 18 V -> 3V3 wiring state (seed 20, `/tmp/ab-seed20-sw09mq2l`) refused with
    `E_POWER_TRANSFER 'regulator': no reviewed source-to-load transfer from '+18V' to
    '+3V3'`: the buck its recipe emits (`AP63203WU-7`) had no reviewed record at all, so
    the only edge the graph held was the power inductor's. These are the real records
    against the real symbol lookup, so a record whose transfer pin names stop resolving
    fails here instead of on a paid run. The placeholder half proves the gate was not
    loosened: an unreviewed converter on the same netlist still has no reviewed path.
    """
    from kicraft.design.part_identity import REVIEWED_PARTS

    bucks = [
        record
        for record in REVIEWED_PARTS
        if "buck" in record.family and record.power_transfer and record.is_portable_candidate
    ]
    assert {record.identity for record in bucks} >= {
        "ap63203wu-7",
        "ap63205wu-7",
        "tlv62569dbvr",
        "tps5430ddar",
        "tps54331ddar",
    }
    architecture = SimpleNamespace(
        requirements=[
            SimpleNamespace(
                id="regulator",
                family="buck-converter",
                role="regulator",
                sheet="POWER",
                obligations=[],
                ports={"input": "VIN", "gnd": "GND", "output": "VOUT"},
                declared_interface=None,
            )
        ]
    )
    for record in bucks:
        converter = SimpleNamespace(
            ref="U1",
            value=record.identity,
            mpn=record.identity,
            symbol=record.symbol,
            footprint=record.footprint,
            sheet="POWER",
        )
        info, _ = validation._pin_info_by_ref(_bom([converter, _part("L1", "4.7uH")], {}))
        vin = validation._pin_number_named(info, "U1", record.port_pins["input"])
        switch = validation._pin_number_named(info, "U1", record.port_pins["switch"])
        gnd = validation._pin_number_named(info, "U1", record.port_pins["ground"])
        assert vin and switch and gnd, record.identity
        bom = _bom(
            [converter, _part("L1", "4.7uH")],
            {
                "VIN": [("U1", vin)],
                "SW_NODE": [("U1", switch), ("L1", "1")],
                "VOUT": [("L1", "2")],
                "GND": [("U1", gnd)],
            },
        )
        result = validation.check_reviewed_power_transfer(architecture, bom)
        assert result.ok, f"{record.identity}: {result.offenders}"

    placeholder = SimpleNamespace(
        ref="U1",
        value="BUCK-3V3",
        mpn="BUCK-3V3",
        symbol="Regulator_Switching:TPS5430DDA",
        footprint="Package_SO:HSOP-8-1EP_3.9x4.9mm_P1.27mm_EP2.41x3.1mm",
        sheet="POWER",
    )
    unreviewed = _bom(
        [placeholder, _part("L1", "4.7uH")],
        {
            "VIN": [("U1", "7")],
            "SW_NODE": [("U1", "8"), ("L1", "1")],
            "VOUT": [("L1", "2")],
            "GND": [("U1", "6")],
        },
    )
    result = validation.check_reviewed_power_transfer(architecture, unreviewed)
    assert not result.ok
    assert any("E_POWER_TRANSFER" in offender for offender in result.offenders)


def test_reviewed_transfer_rejects_distinct_reference_domains(reviewed):
    architecture = SimpleNamespace(requirements=[_conversion_requirement(output_domain="GND_ISO")])
    result = validation.check_reviewed_power_transfer(architecture, _bom([], {}))
    assert not result.ok
    assert result.offenders[0].startswith("E_REFERENCE_DOMAIN")


def test_typed_crossover_rejects_microhenry_magnitude_and_unknown_unit():
    quantity = lambda quantity, value, unit: SimpleNamespace(
        kind="quantitative", quantity=quantity, relation="equal", value=value, unit=unit
    )
    requirement = SimpleNamespace(
        id="crossover",
        family="passive-crossover",
        sheet="XO",
        ports={"input": "AMP", "low_out": "WOOFER", "high_out": "TWEETER"},
        obligations=[quantity("cutoff frequency", 2.5, "kHz"), quantity("load impedance", 8, "ohm")],
    )
    architecture = SimpleNamespace(requirements=[requirement])
    bom = _bom(
        [_part("L1", "510uH", sheet="XO"), _part("C1", "10uF", sheet="XO"), _part("C2", "33uF", sheet="XO")],
        {
            "AMP": [("L1", "1"), ("C1", "1")],
            "WOOFER": [("L1", "2")],
            "HP_MID": [("C1", "2"), ("C2", "1")],
            "TWEETER": [("C2", "2")],
        },
    )
    assert validation.check_typed_passive_crossover_values(architecture, bom).ok
    bom = _bom(
        bom.parts,
        {
            "AMP": [("L1", "1"), ("C1", "1"), ("C2", "1")],
            "WOOFER": [("L1", "2")],
            "TWEETER": [("C1", "2"), ("C2", "2")],
        },
    )
    bom.parts[1].value = "6.8uF"
    bom.parts[2].value = "1uF"
    assert validation.check_typed_passive_crossover_values(architecture, bom).ok

    bom.parts[0].value = "0.51uH"
    result = validation.check_typed_passive_crossover_values(architecture, bom)
    assert not result.ok
    assert "509uH" in result.offenders[0]

    requirement.obligations[0].unit = "MHz"
    result = validation.check_typed_passive_crossover_values(architecture, bom)
    assert not result.ok
    assert "typed equal cutoff" in result.offenders[0]


def _external_one_amp_led_case(
    *, resistor="100m", inductor="22uH", decoupling="10uF", reversed_diode=False,
    open_return=False, counterfeit_load=False, controller_mpn="AL8860", current=1,
):
    driver = SimpleNamespace(
        id="led-current",
        family="constant-current-led-driver",
        sheet="LED",
        ports={"input": "VIN", "gnd": "GND", "set": "LED_A"},
        obligations=[
            SimpleNamespace(kind="quantitative", quantity="LED current", relation="equal", value=current, unit="A"),
            SimpleNamespace(kind="conversion"),
        ],
    )
    output = SimpleNamespace(
        id="external-led",
        family="screw-terminal",
        role="connector",
        sheet="LED",
        ports={"positive": "LED_A", "negative": "LED_K"},
        obligations=[],
    )
    architecture = SimpleNamespace(
        requirements=[driver, output],
        topologies={"LED": "constant current"},
    )
    parts = [
        _part("U3", controller_mpn, mpn=controller_mpn, sheet="LED"),
        _part("L3", inductor, sheet="LED"),
        _part("R3", resistor, sheet="LED"),
        _part("D4", "SS14", sheet="LED"),
        _part("C3", decoupling, sheet="LED"),
    ]
    if counterfeit_load:
        parts.append(_part("D3", "1N4148", sheet="LED"))
    diode_pins = [("D4", "2"), ("C3", "1")]
    switch_pins = [("U3", "5"), ("U3", "6"), ("L3", "2"), ("D4", "1")]
    if reversed_diode:
        diode_pins = [("D4", "1"), ("C3", "1")]
        switch_pins = [("U3", "5"), ("U3", "6"), ("L3", "2"), ("D4", "2")]
    led_cathode = [("L3", "1")]
    if open_return:
        led_cathode = []
    nets = {
        "VIN": [("U3", "8"), ("R3", "1"), *diode_pins],
        "LED_A": [("U3", "1"), ("R3", "2")],
        "LED_K": led_cathode,
        "SW": switch_pins,
        "GND": [("U3", "2"), ("U3", "3"), ("U3", "9"), ("C3", "2")],
    }
    if open_return:
        nets["OPEN_RETURN"] = [("L3", "1")]
    if counterfeit_load:
        nets["LED_A"].append(("D3", "1"))
        nets["LED_K"].append(("D3", "2"))
    return architecture, _bom(parts, nets)


def test_one_amp_led_feedback_accepts_explicit_external_led_loop(reviewed):
    architecture, bom = _external_one_amp_led_case()
    assert validation.check_reviewed_constant_current_led_feedback(architecture, bom).ok


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ({"reversed_diode": True}, "catch diode"),
        ({"open_return": True}, "return inductor"),
        ({"inductor": "0uH"}, "return inductor"),
        ({"resistor": "10m"}, "sense resistor"),
        ({"decoupling": "1uF"}, "input decoupling"),
        ({"counterfeit_load": True}, "reviewed LED physical identity"),
        ({"controller_mpn": "counterfeit-al8860"}, "exactly one reviewed controller"),
        ({"current": 2}, "exceeds reviewed continuous output"),
    ],
)
def test_one_amp_led_feedback_rejects_broken_or_counterfeit_loop(reviewed, case, expected):
    architecture, bom = _external_one_amp_led_case(**case)
    result = validation.check_reviewed_constant_current_led_feedback(architecture, bom)
    assert not result.ok
    assert any(expected in offender for offender in result.offenders)


def test_one_amp_led_feedback_refuses_unknown_reviewed_topology(reviewed, monkeypatch):
    unknown = {**_FACTS[2], "current_feedback": {**_FACTS[2]["current_feedback"], "topology": "unknown"}}
    monkeypatch.setattr(
        validation,
        "_reviewed_fact_for_part",
        lambda part: unknown if part.ref == "U3" else None,
    )
    architecture, bom = _external_one_amp_led_case()
    result = validation.check_reviewed_constant_current_led_feedback(architecture, bom)
    assert not result.ok
    assert "high_side_sense_low_side_switch" in result.offenders[0]


@pytest.mark.parametrize(
    "identity,component_class,accepted",
    [
        ("ESP32-C3-MINI-1-N4", "esp32-c3-module", True),
        ("ESP32-C3-MINI-1-N4", "wireless-module", True),
        ("ESP32-S3-WROOM-1-N8R8", "esp32-c3-module", False),
        # Live cohort 2026-09-30: 11 designs demanded `esp32-s3-module` and every one of them
        # failed 9.42 with "requires 1 real 'esp32-s3-module' physical part(s), found 0" while the
        # construction held the correct S3 module -- the C3 record carries its own class feature
        # (`esp32-c3-module`) and the S3 records carried none. The cross-family cases below are the
        # guard: an S3 module is not a C3 module and a plain ESP32 is not an S3.
        ("ESP32-S3-WROOM-1-N8R8", "esp32-s3-module", True),
        ("ESP32-S3-WROOM-1-N16R8", "esp32-s3-module", True),
        ("ESP32-S3-MINI-1-N8", "esp32-s3-module", True),
        ("ESP32-S3-MINI-1-N8", "esp32-c3-module", False),
        ("ESP32-C3-MINI-1-N4", "esp32-s3-module", False),
        ("ESP32-WROOM-32E-N4", "esp32-s3-module", False),
        # A brief that says only "an ESP32 module" names the family; any reviewed ESP32 satisfies
        # it and a reviewed wifi SoC that is not an ESP32 does not (4 designs demanded it, all 4
        # failed).
        ("ESP32-S3-MINI-1-N8", "esp32-module", True),
        ("ESP32-C3-MINI-1-N4", "esp32-module", True),
        ("NRF52840-QIAA-R", "esp32-module", False),
        ("NRF52840-QIAA-R", "wireless-module", False),
        ("AP63203WU-7", "wireless-module", False),
    ],
)
def test_module_class_requires_the_reviewed_module_not_an_unrelated_ic(identity, component_class, accepted):
    from kicraft.design.part_identity import reviewed_part

    record = reviewed_part(identity)
    part = SimpleNamespace(
        ref="U1", sheet="MCU", value=record.identity, mpn=record.identity,
        symbol=record.symbol, footprint=record.footprint,
    )
    requirement = SimpleNamespace(
        id="mcu", sheet="MCU", exact_part=None, family=component_class,
        declared_interface=None,
        obligations=[SimpleNamespace(kind="physical", component_class=component_class)],
    )
    result = validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=[requirement]), _bom([part], {})
    )
    assert result.ok is accepted
    if not accepted:
        assert all(row.startswith("E_PHYSICAL_REALIZATION") for row in result.offenders)


def test_physical_quantity_requires_exact_reviewed_identity_evidence(monkeypatch):
    record = SimpleNamespace(
        identity="reviewed-bnc",
        family="bnc-connector",
        physical_features=frozenset({"bnc-connector"}),
    )
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda _part: record)
    monkeypatch.setattr(validation, "_pin_info_by_ref", lambda _bom: ({}, {}))
    requirement = SimpleNamespace(
        id="bnc",
        sheet="IO",
        exact_part=None,
        family="bnc-connector",
        declared_interface=None,
        obligations=[
            SimpleNamespace(kind="physical", component_class="bnc-connector"),
            SimpleNamespace(kind="quantity", subject="bnc-connector", minimum=2),
        ],
    )
    architecture = SimpleNamespace(requirements=[requirement])
    bom = _bom([_part("J1", "BNC", sheet="IO"), _part("J2", "BNC", sheet="IO")], {})
    assert validation.check_requirement_physical_realization(architecture, bom).ok

    result = validation.check_requirement_physical_realization(architecture, _bom(bom.parts[:1], {}))
    assert not result.ok
    assert result.offenders[0].startswith("E_PHYSICAL_REALIZATION")


def test_a_class_with_no_reviewed_carrier_is_not_realized_by_a_reviewed_part_of_another_class(
    tmp_path, monkeypatch
):
    """A coverage gap stays refused: known hardware must not become an unrelated component.

    `solder-jumper` names a part the library carries no record for -- the CAN-termination
    reference asks for one (`SolderJumper_2_Open`) and the reviewed header/connector records are
    not that class -- so a reviewed part of another class must not satisfy the demand, and the
    gate reports the class as uncovered ("real" evidence) rather than pretending the library can
    answer it.
    """
    from kicraft.design.part_identity import realizable_physical_features

    monkeypatch.setenv("KICRAFT_RESEARCHED_RECORDS", str(tmp_path / "researched.json"))
    assert realizable_physical_features("solder-jumper") == frozenset()
    stepper = _part("U1", "A4988SETTR-T", mpn="A4988SETTR-T")
    stepper.symbol = "a4988:A4988SETTR-T"
    stepper.footprint = "a4988:WQFN-28_L5.0-W5.0-P0.50-BL-EP3.2"
    requirement = SimpleNamespace(
        id="motor_a",
        sheet="POWER",
        exact_part=None,
        family="solder-jumper",
        declared_interface=None,
        obligations=[
            SimpleNamespace(kind="physical", component_class="solder-jumper"),
        ],
    )
    result = validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=[requirement]), _bom([stepper], {})
    )
    assert not result.ok
    assert all(row.startswith("E_PHYSICAL_REALIZATION") for row in result.offenders)
    assert "requires 1 real 'solder-jumper'" in result.offenders[0]


def test_a_recipe_realized_regulator_is_a_reviewed_transfer_path():
    """A curated recipe is reviewed evidence: §9.39 must read its declared input/output ports.

    Live seed-43 run (2026-09-26): an AMS1117 board -- a curated recipe, the sanctioned way to
    build a converter -- was refused with "no reviewed source-to-load transfer from
    'VIN_PROTECTED' to '+3V3'" because §9.39 read only the reviewed *records*' `power_transfer`
    mapping, and a recipe-realized part has no record at all.
    """
    from types import SimpleNamespace

    from kicraft.design.synthesis.validation import _recipe_transfer_pairs

    requirement = SimpleNamespace(
        id="reg", ports={"input": "VIN_PROTECTED", "output": "+3V3", "gnd": "GND"}
    )
    bom = SimpleNamespace(
        recipe_ownership=[SimpleNamespace(recipe="ams1117-3v3@1", requirement_ids=("reg",))]
    )
    assert _recipe_transfer_pairs(requirement, bom) == [("VIN_PROTECTED", "+3V3")]

    # A requirement the manifest does not own, or an unknown recipe, contributes nothing.
    assert _recipe_transfer_pairs(requirement, SimpleNamespace(recipe_ownership=[])) == []
    assert (
        _recipe_transfer_pairs(
            requirement,
            SimpleNamespace(
                recipe_ownership=[
                    SimpleNamespace(recipe="not-a-recipe@9", requirement_ids=("reg",))
                ]
            ),
        )
        == []
    )


def test_a_bundle_part_owns_a_declared_interface(monkeypatch):
    """A vendored part with no MPN is the identity its reviewed pair names, not a missing part.

    Live seed-43 run (2026-09-26): the power LED -- a vendored bundle, no MPN -- was reported as
    "needs exactly one identity-matched BOM component with resolved pin inventory", which hid the
    real complaint (its declared pin sat on the wrong net).
    """
    from types import SimpleNamespace

    record = SimpleNamespace(
        identity="warm-white-led",
        family="warm-white-led",
        physical_features=frozenset({"warm-white-led"}),
    )
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda _part: record)
    monkeypatch.setattr(
        validation,
        "_pin_info_by_ref",
        lambda _bom: ({"D1": {"1": {"name": "K"}, "2": {"name": "A"}}}, {}),
    )
    requirement = SimpleNamespace(
        id="power_led",
        sheet="POWER INDICATOR",
        role="user_io",
        family="warm-white-led",
        exact_part=None,
        obligations=[],
        declared_interface=SimpleNamespace(
            ports=[SimpleNamespace(key="drive", pin="1", pin_selector=None, pin_name=None)]
        ),
        ports={"drive": "+3V3"},
    )
    architecture = SimpleNamespace(requirements=[requirement], power_nets=["+3V3"])
    led = SimpleNamespace(
        ref="D1", value="E6C0805WWAY1UDA", mpn=None, symbol="e6c0805wway1uda:L",
        footprint="e6c0805wway1uda:LED", sheet="POWER INDICATOR",
    )
    bom = SimpleNamespace(
        parts=[led],
        connections=[
            SimpleNamespace(net_name="+3V3", endpoints=[SimpleNamespace(ref="D1", pin="2")])
        ],
        recipe_ownership=[],
    )

    result = validation.check_requirement_physical_realization(
        architecture, bom, declared_interface_scope="model_owned"
    )
    # The complaint is now the real one (the claimed pin is not on the claimed net), not
    # "needs exactly one identity-matched component".
    assert not result.ok
    assert "E_DECLARED_INTERFACE" in result.offenders[0]
    assert "identity-matched" not in result.offenders[0]


def test_a_declared_pin_on_the_drive_path_past_a_series_element_is_satisfied(monkeypatch):
    """A rail-driven indicator: the pin is driven from the rail *through* its resistor.

    Live seed-43 run (2026-09-26): the architecture declares the power LED's `drive` port on the
    3.3 V rail with pin 1 as its declared contact; the correct wiring is rail -> 249 ohm resistor
    -> anode (pin 2) -> cathode (pin 1) -> ground. Comparing net names alone called that a defect
    ("expected '+3V3' on declared pin selector '1' of D1, found 'GND'") although the circuit is
    exactly right -- owner: *"you could easily satisfy that if you wanted by moving the series
    resistor to after the LED but it doesnt matter … our system of checks … is erroring on a valid
    design over semantics"*.
    """
    from types import SimpleNamespace

    led = SimpleNamespace(
        ref="D1", value="E6C0805WWAY1UDA", mpn=None, symbol="led:L", footprint="led:LED",
        sheet="POWER INDICATOR", resolution_id="bom-s004",
    )
    resistor = SimpleNamespace(
        ref="R2", value="249", mpn="RC0805FR-07249RL", symbol="Device:R",
        footprint="Resistor_SMD:R_0608Metric", sheet="POWER INDICATOR", resolution_id="bom-s004",
    )
    records = {
        "D1": SimpleNamespace(identity="warm-white-led", family="warm-white-led",
                              physical_features=frozenset({"warm-white-led"}), contacts=("1", "2")),
        "R2": SimpleNamespace(identity="chip-resistor", family="chip-resistor",
                              physical_features=frozenset({"chip-resistor"}), contacts=("1", "2")),
    }
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda part: records[part.ref])
    monkeypatch.setattr(
        validation,
        "_pin_info_by_ref",
        lambda _bom: ({"D1": {"1": {"name": "C"}, "2": {"name": "A"}},
                       "R2": {"1": {"name": "1"}, "2": {"name": "2"}}}, {}),
    )
    requirement = SimpleNamespace(
        id="power_led", sheet="POWER INDICATOR", role="user_io", family="warm-white-led",
        # Named, exactly as the architecture names it: the requirement's interface is the LED,
        # not every part on its sheet -- without this the series resistor reads as a second
        # declared interface.
        exact_part="warm-white-led",
        obligations=[],
        declared_interface=SimpleNamespace(
            ports=[SimpleNamespace(key="drive", pin="1", pin_selector=None, pin_name=None)]
        ),
        ports={"drive": "+3V3"},
    )
    architecture = SimpleNamespace(requirements=[requirement], power_nets=["+3V3", "GND"])

    def bom_with(*connections):
        return SimpleNamespace(
            parts=[led, resistor], connections=list(connections), recipe_ownership=[]
        )

    # The LED lit through its resistor: ground -> cathode, anode -> resistor -> rail.
    lit = bom_with(
        SimpleNamespace(net_name="+3V3", endpoints=[SimpleNamespace(ref="R2", pin="1")]),
        SimpleNamespace(
            net_name="LED_A",
            endpoints=[SimpleNamespace(ref="R2", pin="2"), SimpleNamespace(ref="D1", pin="2")],
        ),
        SimpleNamespace(net_name="GND", endpoints=[SimpleNamespace(ref="D1", pin="1")]),
    )
    assert validation.check_requirement_physical_realization(
        architecture, lit, declared_interface_scope="model_owned"
    ).ok

    # A declared pin with no path of its own to the port's net still fails.
    wrong = bom_with(
        SimpleNamespace(net_name="+3V3", endpoints=[SimpleNamespace(ref="R2", pin="1")]),
        SimpleNamespace(net_name="LED_A", endpoints=[SimpleNamespace(ref="R2", pin="2")]),
        SimpleNamespace(net_name="RESET_N", endpoints=[SimpleNamespace(ref="D1", pin="2")]),
    )
    refused = validation.check_requirement_physical_realization(
        architecture, wrong, declared_interface_scope="model_owned"
    )
    assert not refused.ok
    assert "E_DECLARED_INTERFACE" in refused.offenders[0]


def test_new_part_category_is_realized_by_a_resolved_part(monkeypatch):
    """A class the reviewed library has never covered is proven by a real resolved part.

    Decision 2026-09-18: the library can only answer for the classes it covers, so a demand
    for a new part category (`gps-module`) is met by a real, resolvable part — exact MPN,
    resolvable symbol pin inventory, footprint — instead of refusing every design that needs
    hardware nobody has reviewed yet. A label without an orderable identity is not evidence.
    """
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda _part: None)
    monkeypatch.setattr(validation, "_pin_info_by_ref", lambda _bom: ({}, {}))
    requirement = SimpleNamespace(
        id="gnss",
        sheet="IO",
        exact_part=None,
        family="gnss-receiver",
        declared_interface=None,
        obligations=[SimpleNamespace(kind="physical", component_class="gps-module")],
    )
    architecture = SimpleNamespace(requirements=[requirement])
    resolved = SimpleNamespace(
        ref="U1", value="NEO-6M", mpn="NEO-6M-0-001", symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric", sheet="IO",
    )
    assert validation.check_requirement_physical_realization(architecture, _bom([resolved], {})).ok

    unlabelled = SimpleNamespace(**{**vars(resolved), "mpn": None})
    result = validation.check_requirement_physical_realization(architecture, _bom([unlabelled], {}))
    assert not result.ok
    assert result.offenders[0].startswith("E_PHYSICAL_REALIZATION")
    assert "requires 1 real 'gps-module'" in result.offenders[0]


def test_board_fact_obligations_are_not_physical_component_demands(monkeypatch):
    """§9.42 reads `physical` rows only: a board feature and an absence demand no part.

    The canary (2026-09-17, `led-cc-driver`, `star-ornament`, `buck-3a`, `thermocouple-amp`)
    recorded "printed copper area as a heatsink" and "no microcontroller" as physical classes, so
    this check refused them forever. A `negative` row naming a class the BOM *does* contain is
    neither satisfied nor unfulfilled by it.
    """
    record = SimpleNamespace(
        identity="reviewed-bnc",
        family="bnc-connector",
        physical_features=frozenset({"bnc-connector"}),
    )
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda _part: record)
    monkeypatch.setattr(validation, "_pin_info_by_ref", lambda _bom: ({}, {}))
    requirement = SimpleNamespace(
        id="led_driver",
        sheet="IO",
        exact_part=None,
        family="led-cc-driver",
        declared_interface=None,
        obligations=[
            SimpleNamespace(kind="fabrication", feature="copper-area", minimum=300, unit="mm2"),
            SimpleNamespace(kind="negative", absent_class="bnc-connector"),
        ],
    )
    architecture = SimpleNamespace(requirements=[requirement])
    bom = _bom([_part("J1", "BNC", sheet="IO")], {})

    assert validation.check_requirement_physical_realization(architecture, bom).ok


def test_a_count_two_requirements_share_is_one_demand_on_the_sheet(monkeypatch):
    """Two requirements carrying the brief's "two connectors" row demand two parts, not four.

    Live seed-43 commit (2026-09-25) refused the board with ``E_PHYSICAL_REALIZATION 'JST
    INTERFACES'/'jst-xh-connector': jst1×2, jst2×2 demand 4 distinct part(s), but only 2 exact
    reviewed MPN/symbol/footprint realization(s) exist`` for a sheet holding exactly the two
    connectors the brief asked for: the shared row was summed once per requirement.
    """
    record = SimpleNamespace(
        identity="reviewed-xh",
        family="jst-xh-connector",
        physical_features=frozenset({"jst-xh-connector"}),
    )
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda _part: record)
    monkeypatch.setattr(validation, "_pin_info_by_ref", lambda _bom: ({}, {}))

    def requirement(ident: str):
        return SimpleNamespace(
            id=ident,
            sheet="XH",
            exact_part=None,
            family="jst-xh-connector",
            declared_interface=None,
            obligations=[
                SimpleNamespace(kind="physical", original_obligation_id="xh",
                                component_class="jst-xh-connector"),
                SimpleNamespace(kind="quantity", original_obligation_id="two_xh",
                                subject="jst-xh connectors", minimum=2),
            ],
        )

    architecture = SimpleNamespace(requirements=[requirement("jst1"), requirement("jst2")])
    two = _bom([_part("J1", "XH", sheet="XH"), _part("J2", "XH", sheet="XH")], {})
    assert validation.check_requirement_physical_realization(architecture, two).ok

    one = _bom([_part("J1", "XH", sheet="XH")], {})
    result = validation.check_requirement_physical_realization(architecture, one)
    assert not result.ok
    assert any("demand 2 distinct" in offender for offender in result.offenders)


def _terminal_requirement(ident: str, *, minimum: int, exact_part: str | None = None):
    """One screw-terminal requirement carrying the brief's shared terminal-count row."""
    return SimpleNamespace(
        id=ident,
        sheet="ANALOG INPUTS",
        exact_part=exact_part,
        family="screw-terminal",
        declared_interface=None,
        obligations=[
            SimpleNamespace(kind="physical", original_obligation_id="screw-terminals",
                            component_class="screw-terminal"),
            SimpleNamespace(kind="quantity", original_obligation_id="eight-input-terminals",
                            subject="screw-terminal", minimum=minimum),
        ],
    )


def _reviewed_terminal_part(ref: str, identity: str):
    """A BOM part carrying a reviewed terminal record's own symbol/footprint/mpn."""
    from kicraft.design.part_identity import reviewed_part

    record = reviewed_part(identity)
    assert record is not None
    return SimpleNamespace(
        ref=ref, value=record.identity, mpn=record.identity,
        symbol=record.symbol, footprint=record.footprint, sheet="ANALOG INPUTS",
        datasheet=None, sourcing_note=None,
    )


def _generic_terminal_part(ref: str, contacts: int):
    """A BOM part carrying the stock multi-position block the lowerer emits."""
    return SimpleNamespace(
        ref=ref, value=f"ScrewTerminal_1x{contacts:02d}", mpn=None,
        symbol=f"Connector:Screw_Terminal_01x{contacts:02d}",
        footprint=("TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-"
                   f"{contacts}_1x{contacts:02d}_P5.00mm_Horizontal"),
        sheet="ANALOG INPUTS", datasheet=None, sourcing_note=None,
    )


def test_a_multi_position_terminal_realizes_one_terminal_per_contact():
    """A nine-contact terminal demand is met by the contacts the reviewed blocks publish.

    The held-out analog briefs demand "eight single-ended analog inputs plus ground on screw
    terminals" and the reviewed terminal vocabulary carries only multi-position blocks
    (WJ126V-5.0-{02,03,04}P, KiCad-stock Phoenix MKDS-1,5 1x02..1x12). Counting parts made the
    recorded r2 commit unrepairable: ``terminals_a×9, terminals_b×9, terminals_ground×9 demand
    9 distinct part(s), but only 3 exact reviewed MPN/symbol/footprint realization(s) exist``
    for a board holding 4+4+2 positions. The count is in contacts, so the nine-position block
    and the 4+4+2 composition pass, and an eight-contact design is still refused.
    """
    nine = _terminal_requirement("terminals", minimum=9)
    assert validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=[nine]), _bom([_generic_terminal_part("J1", 9)], {})
    ).ok

    short = validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=[nine]), _bom([_generic_terminal_part("J1", 8)], {})
    )
    assert not short.ok
    assert any(
        "demand 9 distinct contact(s), but only 8 exact reviewed" in row
        for row in short.offenders
    )

    requirements = [
        _terminal_requirement("terminals_a", minimum=9, exact_part="WJ126V-5.0-04P-14-00A"),
        _terminal_requirement("terminals_b", minimum=9, exact_part="WJ126V-5.0-04P-14-00A"),
        _terminal_requirement("terminals_ground", minimum=9, exact_part="WJ126V-5.0-02P-14-00A"),
    ]
    composed = _bom(
        [
            _reviewed_terminal_part("J1", "wj126v-5.0-04p-14-00a"),
            _reviewed_terminal_part("J2", "wj126v-5.0-04p-14-00a"),
            _reviewed_terminal_part("J3", "wj126v-5.0-02p-14-00a"),
        ],
        {},
    )
    assert validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=requirements), composed
    ).ok

    # The same composition one contact short of the demand is still refused.
    short_composition = _bom(
        [
            _reviewed_terminal_part("J1", "wj126v-5.0-04p-14-00a"),
            _reviewed_terminal_part("J2", "wj126v-5.0-04p-14-00a"),
        ],
        {},
    )
    result = validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=requirements[:2]), short_composition
    )
    assert not result.ok
    assert any("demand 9 distinct contact(s), but only 8" in row for row in result.offenders)

    # The STM32 brief's sixteen inputs plus ground (seventeen contacts) is the same shape one
    # size up: 4+4+4+4+2 reviewed positions meet it, and four four-position blocks do not.
    sixteen = [
        _terminal_requirement(f"terminals_{suffix}", minimum=17,
                              exact_part=f"WJ126V-5.0-{contacts:02d}P-14-00A")
        for suffix, contacts in (("a", 4), ("b", 4), ("c", 4), ("d", 4), ("ground", 2))
    ]
    full = _bom(
        [_reviewed_terminal_part(f"J{index}", identity) for index, identity in enumerate(
            ["wj126v-5.0-04p-14-00a"] * 4 + ["wj126v-5.0-02p-14-00a"], 1
        )],
        {},
    )
    assert validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=sixteen), full
    ).ok
    assert not validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=sixteen[:4]), _bom(full.parts[:4], {})
    ).ok

    # One requirement may own the whole composition: the pinned reviewed block is the unit,
    # so an eight-contact requirement realized by two four-position blocks of that same part
    # is eight contacts, and a single block is four. Held-out r3 `rp2040-dual-adc-usb`
    # (2026-09-30) declared exactly that shape in one `terminals` requirement.
    one = _terminal_requirement("terminals", minimum=8, exact_part="WJ126V-5.0-04P-14-00A")
    assert validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=[one]),
        _bom(
            [
                _reviewed_terminal_part("J1", "wj126v-5.0-04p-14-00a"),
                _reviewed_terminal_part("J2", "wj126v-5.0-04p-14-00a"),
            ],
            {},
        ),
    ).ok
    single_block = validation.check_requirement_physical_realization(
        SimpleNamespace(requirements=[one]),
        _bom([_reviewed_terminal_part("J1", "wj126v-5.0-04p-14-00a")], {}),
    )
    assert not single_block.ok
    assert any("demand 8 distinct contact(s), but only 4" in row for row in single_block.offenders)


def test_distinct_requirement_owners_cannot_share_one_reviewed_connector(monkeypatch):
    record = SimpleNamespace(
        identity="reviewed-bnc",
        family="bnc-connector",
        physical_features=frozenset({"bnc-connector"}),
    )
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda _part: record)
    monkeypatch.setattr(validation, "_pin_info_by_ref", lambda _bom: ({}, {}))
    requirement = lambda ident: SimpleNamespace(
        id=ident,
        sheet="IO",
        exact_part=None,
        family="bnc-connector",
        declared_interface=None,
        obligations=[SimpleNamespace(kind="physical", component_class="bnc-connector")],
    )
    architecture = SimpleNamespace(requirements=[requirement("input"), requirement("output")])
    bom = _bom([_part("J1", "BNC", sheet="IO")], {})
    result = validation.check_requirement_physical_realization(architecture, bom)
    assert not result.ok
    assert any("demand 2 distinct" in offender for offender in result.offenders)


def _declared_connector_case(*, second_symbol: str = "usb-a:SMD"):
    """Two identical USB-A outputs behind one model-declared interface family."""

    def declared(ident, net):
        return SimpleNamespace(
            id=ident,
            sheet="POWER",
            exact_part="U-A-24SS-W-2",
            family="usb-a-power-output",
            declared_interface=SimpleNamespace(
                ports=[SimpleNamespace(key="vbus", pin="1", pin_selector=None, pin_name=None)]
            ),
            obligations=[],
            ports={"vbus": net},
        )

    parts = [
        SimpleNamespace(
            ref="J2",
            value="U-A-24SS-W-2",
            mpn="U-A-24SS-W-2",
            symbol="usb-a:SMD",
            footprint="usb-a:SMD",
            sheet="POWER",
        ),
        SimpleNamespace(
            ref="J3",
            value="U-A-24SS-W-2",
            mpn="U-A-24SS-W-2",
            symbol=second_symbol,
            footprint="usb-a:SMD",
            sheet="POWER",
        ),
    ]
    architecture = SimpleNamespace(requirements=[declared("port1", "USB_A1_5V"), declared("port2", "USB_A2_5V")])
    bom = _bom(parts, {"USB_A1_5V": [("J2", "1")], "USB_A2_5V": [("J3", "1")]})
    bom.recipe_ownership = []
    return architecture, bom


def test_identical_declared_interface_instances_are_one_owned_family(monkeypatch):
    """A bank of identical instances realizes its own port-to-pin map.

    The splitter reference ships two USB-A receptacles and two load switches; each
    declared interface must be checked against one instance of its own identity
    family, not refused because the family has two instances.
    """
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda _part: None)
    monkeypatch.setattr(
        validation,
        "_pin_info_by_ref",
        lambda _bom: ({"J2": {"1": {"name": "VBUS"}}, "J3": {"1": {"name": "VBUS"}}}, {}),
    )
    architecture, bom = _declared_connector_case()
    assert validation.check_requirement_physical_realization(architecture, bom).ok

    # A family whose members are different hardware is still split ownership.
    architecture, bom = _declared_connector_case(second_symbol="other-vendor:USB-A")
    result = validation.check_requirement_physical_realization(architecture, bom)
    assert not result.ok
    assert any("needs exactly one identity-matched BOM component" in offender for offender in result.offenders)


def test_declared_interface_requires_a_wired_instance(monkeypatch):
    """One instance on the wrong net is still an unrealized declared interface."""
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda _part: None)
    monkeypatch.setattr(
        validation,
        "_pin_info_by_ref",
        lambda _bom: ({"J2": {"1": {"name": "VBUS"}}, "J3": {"1": {"name": "VBUS"}}}, {}),
    )
    architecture, bom = _declared_connector_case()
    bom = _bom(bom.parts, {"GND": [("J2", "1")], "USB_A2_5V": [("J3", "1")]})
    bom.recipe_ownership = []
    result = validation.check_requirement_physical_realization(architecture, bom)
    assert not result.ok
    assert any("expected 'USB_A1_5V'" in offender for offender in result.offenders)


def test_frozen_pd_and_placeholder_converter_fail_without_new_conversion_obligations(reviewed):
    pd = SimpleNamespace(
        id="pd",
        family="usb-pd-fixed-trigger",
        role="power_input",
        sheet="PD",
        obligations=[],
        ports={"vbus": "VBUS", "gnd": "GND"},
        declared_interface=None,
    )
    dual = SimpleNamespace(
        id="dual",
        family="dual-output-dc-dc-converter",
        role="regulator",
        sheet="POWER",
        obligations=[],
        ports={"input": "VIN", "gnd": "GND", "positive": "+12V", "negative": "-12V"},
        declared_interface=None,
    )
    architecture = SimpleNamespace(
        requirements=[pd, dual],
        power_nets=["VBUS", "VBUS_NEGOTIATED", "VIN", "+12V", "-12V", "GND"],
    )
    result = validation.check_reviewed_power_transfer(architecture, _bom([], {}))
    assert not result.ok
    assert any("E_POWER_TRANSFER 'pd'" in offender for offender in result.offenders)
    assert any("E_POWER_TRANSFER 'dual'" in offender for offender in result.offenders)


def _isolated_converter_case(*, output_domain="0V_ISO"):
    requirement = SimpleNamespace(
        id="converter",
        family="dual-output-dc-dc-converter",
        role="regulator",
        sheet="POWER",
        obligations=[SimpleNamespace(kind="conversion")],
        ports={
            "input_positive": "VIN",
            "input_return": "GND",
            "output_positive": "+12V",
            "output_negative": "-12V",
            "output_common": "0V_ISO",
        },
        declared_interface=SimpleNamespace(
            ports=[
                SimpleNamespace(key="input_positive", reference_domain=None),
                SimpleNamespace(key="input_return", reference_domain="GND"),
                SimpleNamespace(key="output_common", reference_domain=output_domain),
            ]
        ),
    )
    return SimpleNamespace(requirements=[requirement]) 


def test_isolated_converter_domains_come_from_each_sides_own_return_port(reviewed):
    """A rail-bound port cannot also carry its return domain.

    The derivation refuses two nets on one port (`conflicting_port_binding`), so an
    isolated converter declares each side's domain on that side's own return port
    (`input_return` / `output_common`). Those published names are what §9.39 reads.
    """
    pins = {
        "VIN": {"name": "VIN", "type": "power_in"},
        "GND": {"name": "GND", "type": "power_in"},
        "+VO": {"name": "+VO", "type": "power_out"},
        "-VO": {"name": "-VO", "type": "power_out"},
        "0V": {"name": "0V", "type": "passive"},
    }
    original = validation._pin_info_by_ref
    validation._pin_info_by_ref = lambda _bom: ({"U4": pins}, {})
    try:
        bom = _bom(
            [_part("U4", "WRA2412S-3WR2", mpn="WRA2412S-3WR2")],
            {
                "VIN": [("U4", "VIN")],
                "GND": [("U4", "GND")],
                "+12V": [("U4", "+VO")],
                "-12V": [("U4", "-VO")],
                "0V_ISO": [("U4", "0V")],
            },
        )
        assert validation.check_reviewed_power_transfer(_isolated_converter_case(), bom).ok

        # One shared return is not an isolated domain model.
        shared = validation.check_reviewed_power_transfer(
            _isolated_converter_case(output_domain="GND"), bom
        )
        assert not shared.ok
        assert any("E_REFERENCE_DOMAIN" in offender for offender in shared.offenders)
    finally:
        validation._pin_info_by_ref = original


def test_demanded_class_aliases_apply_in_the_realization_gate_too(monkeypatch):
    """One alias vocabulary for both gates that read the reviewed features.

    `three-position-selector-switch` (the class `usb-pd-trigger` demands) matched neither the BOM
    work-unit check nor this §9.42 gate, whose offender read `requires 1 reviewed
    'three-position-selector-switch' physical part(s), found 0` while the reviewed record for the
    part spells the class `three-position-selector`. A class aliased in one gate and unknown in the
    other is a vocabulary bug, not a model error.
    """
    from kicraft.design.part_identity import canonical_physical_features

    assert canonical_physical_features("three-position-selector-switch") == frozenset(
        {"three-position-selector", "sp3t-selector"}
    )
    record = SimpleNamespace(
        identity="ss13d07vg4",
        family="three-position-selector",
        physical_features=frozenset({"three-position-selector", "sp3t-selector"}),
    )
    monkeypatch.setattr(validation, "_reviewed_identity_for_bom_part", lambda _part: record)
    monkeypatch.setattr(validation, "_pin_info_by_ref", lambda _bom: ({}, {}))
    requirement = SimpleNamespace(
        id="dc_output",
        sheet="OUTPUT",
        exact_part=None,
        family="voltage-selector-switch",
        declared_interface=None,
        obligations=[
            SimpleNamespace(kind="physical", component_class="three-position-selector-switch")
        ],
    )
    architecture = SimpleNamespace(requirements=[requirement])
    bom = _bom([_part("SW1", "SS13D07VG4", sheet="OUTPUT")], {})

    assert validation.check_requirement_physical_realization(architecture, bom).ok


def test_a_researched_records_catalog_supply_range_is_not_an_input_declaration(monkeypatch):
    """A researched part's own supply range must not become a false input-range refusal.

    §9.38 refuses a record that states a voltage input without the pin it lands on, because it
    then cannot compare the rail. A researched record declares no pin, so a catalog supply range
    claimed under the input-domain names would turn every researched regulator into a build
    failure; the range is claimed as ``supply_min_v``/``supply_max_v`` instead -- the names the
    hand-reviewed MCU records use -- which this check reads only when the record also names its
    supply port.
    """
    from kicraft.design import part_identity, part_research

    record = part_research.build_record(
        "linear-regulator",
        {"lcsc": "C1", "description": "a 3.3 V linear regulator"},
        {"name": "ldo-c1", "mpn": "XC6206", "sourcing": {"lcsc": "C1"}},
        attributes={
            "Voltage - Supply": "2.6V~6V",
            "Output Voltage": "3.3V",
            "Output Current": "200mA",
        },
    )
    part = part_identity._reviewed_part_from_record(record)
    assert (part.operating_limits["supply_min_v"], part.operating_limits["supply_max_v"]) == (2.6, 6.0)
    assert part.operating_limits["output_current_a"] == 0.2

    monkeypatch.setattr(validation, "_reviewed_fact_for_part", lambda _part: vars(part))
    monkeypatch.setattr(
        validation,
        "_pin_info_by_ref",
        lambda _bom: ({"U9": {"1": {"name": "VIN", "type": "power_in"}}}, {}),
    )
    bom = _bom([_part("U9", "XC6206", mpn="XC6206")], {"+5V": [("U9", "1")]})
    architecture = SimpleNamespace(rail_voltages={"+5V": 5.0})

    assert validation.check_reviewed_input_operating_ranges(architecture, bom).ok


def test_motor_supply_over_rating_is_refused_on_the_reviewed_vm_domain(reviewed):
    """A part's own supply rating decides, whatever domain the record spells it in.

    The DRV8833 publishes ``motor_supply_min_v``/``motor_supply_max_v`` (2.7–10.8 V) against its
    ``vm`` pin and no ``vin``/``input`` character at all, so this check used to skip the record as
    "no voltage input to compare" and an 18 V rail reached the board. Surprise-me seed 37 wires
    exactly that: rail ``VIN_18V`` at 18 V into the driver's supply.
    """
    bom = _bom([_part("U4", "DRV8833PWPR", mpn="DRV8833PWPR")], {"+18V": [("U4", "12")]})
    architecture = SimpleNamespace(rail_voltages={"+18V": 18.0})

    result = validation.check_reviewed_input_operating_ranges(architecture, bom)
    assert not result.ok
    assert "E_INPUT_OPERATING_RANGE" in result.offenders[0]
    assert "MOTOR SUPPLY range 2.7–10.8V" in result.offenders[0]

    # The same part on a rail inside its motor rating stays clean.
    assert validation.check_reviewed_input_operating_ranges(
        SimpleNamespace(rail_voltages={"+18V": 9.0}), bom
    ).ok


@pytest.mark.parametrize(
    ("positive", "negative", "rating", "accepted"),
    [(18.0, 0.0, "10V", False), (18.0, 0.0, "25V", True),
     (12.0, -12.0, "16V", False), (12.0, 5.0, "10V", True),
     (10.0, 0.0, "10V", True), (18.0, 0.0, None, False)],
)
def test_selected_capacitor_rating_covers_actual_terminal_difference(
    reviewed, monkeypatch, positive, negative, rating, accepted
):
    from kicraft.parts_library import jlcparts

    monkeypatch.setattr(
        jlcparts, "parameters",
        lambda cid: {"attributes": {"Voltage Rating": rating} if rating else {}},
    )
    cap = _part("C1", "10uF")
    cap.sourcing_note = "LCSC C19702"
    bom = _bom([cap], {"POS": [("C1", "1")], "RETURN": [("C1", "2")]})
    result = validation.check_reviewed_input_operating_ranges(
        SimpleNamespace(rail_voltages={"POS": positive, "RETURN": negative}), bom,
    )
    assert result.ok is accepted
    if not accepted:
        assert any("E_CAPACITOR_VOLTAGE C1" in row for row in result.offenders)


def test_same_capacitor_identity_does_not_share_voltage_stress_between_rails(
    reviewed, monkeypatch,
):
    from kicraft.parts_library import jlcparts

    monkeypatch.setattr(
        jlcparts, "parameters", lambda cid: {"attributes": {"Voltage Rating": "10V"}},
    )
    parts = [_part("C1", "10uF"), _part("C2", "10uF")]
    for part in parts:
        part.sourcing_note = "LCSC C19702"
    bom = _bom(parts, {
        "LOW": [("C1", "1")], "HIGH": [("C2", "1")],
        "GND": [("C1", "2"), ("C2", "2")],
    })
    result = validation.check_reviewed_input_operating_ranges(
        SimpleNamespace(rail_voltages={"LOW": 3.3, "HIGH": 18.0, "GND": 0.0}), bom,
    )
    assert not result.ok
    assert all("C2" in row for row in result.offenders)
    assert any("18V" in row for row in result.offenders)


def _nonsynchronous_buck_support(*, diode_identity="SS34"):
    """Independent TI SLVS839H support circuit; not generated from a recipe."""
    from kicraft.design.models import BomPart

    parts = [
        BomPart(
            ref="U1", value="TPS54331DDAR", mpn="TPS54331DDAR", sheet="POWER",
            symbol="tps54331:TPS54331DDAR",
            footprint="tps54331:SOIC-8_L4.9-W3.9-P1.27-LS6.0-BL-EP",
        ),
        BomPart(
            ref="C1", value="100nF", sheet="POWER",
            symbol="Device:C", footprint="Capacitor_SMD:C_0603_1608Metric",
        ),
        BomPart(
            ref="D1", value=diode_identity, mpn=diode_identity, sheet="POWER",
            symbol="Device:D_Schottky", footprint="Diode_SMD:D_SMA",
        ),
    ]
    bom = _bom(parts, {
        "VIN": [("U1", "2")],
        "BOOT": [("U1", "1"), ("C1", "1")],
        "PH": [("U1", "8"), ("C1", "2"), ("D1", "1")],
        "GND": [("U1", "7"), ("U1", "9"), ("D1", "2")],
    })
    bom.no_connect_pins = [SimpleNamespace(ref="U1", pin="3")]
    return bom


@pytest.mark.parametrize("identity", ["SS34", "B340A"])
def test_nonsynchronous_buck_accepts_reviewed_equivalent_rectifiers(identity):
    bom = _nonsynchronous_buck_support(diode_identity=identity)
    assert validation.check_reviewed_device_support_networks(bom).ok


@pytest.mark.parametrize("defect", ["missing", "reversed", "unrated"])
def test_nonsynchronous_buck_rejects_missing_or_unproven_freewheel_path(defect):
    bom = _nonsynchronous_buck_support()
    if defect == "missing":
        bom.parts = [part for part in bom.parts if part.ref != "D1"]
    elif defect == "reversed":
        for connection in bom.connections:
            for endpoint in connection.endpoints:
                if endpoint.ref == "D1":
                    endpoint.pin = "2" if endpoint.pin == "1" else "1"
    else:
        diode = next(part for part in bom.parts if part.ref == "D1")
        diode.mpn = diode.value = "unreviewed Schottky"
    result = validation.check_reviewed_device_support_networks(bom)
    assert not result.ok
    assert any("E_CATCH_DIODE_SUPPORT" in offender for offender in result.offenders)


@pytest.mark.parametrize(
    ("enable_voltage", "accepted"),
    [(None, True), (3.3, True), (5.0, True), (0.0, False), (12.0, False)],
)
def test_reviewed_enable_respects_floating_mode_threshold_and_absolute_maximum(
    enable_voltage, accepted,
):
    bom = _nonsynchronous_buck_support()
    rails = {"VIN": 5.0, "GND": 0.0}
    if enable_voltage is not None:
        rails["ENABLE"] = enable_voltage
        bom.no_connect_pins = []
        bom.connections.append(SimpleNamespace(
            net_name="ENABLE", endpoints=[SimpleNamespace(ref="U1", pin="3")],
        ))
    assert validation.check_reviewed_input_operating_ranges(
        SimpleNamespace(rail_voltages=rails), bom,
    ).ok is accepted


@pytest.mark.parametrize(("top", "bottom", "accepted"), [
    ("511k", "100k", False),
    ("10k", "10k", True),
])
def test_usb_powered_buck_enable_bias_does_not_rely_on_typical_pullup_current(
    top, bottom, accepted,
):
    from kicraft.design.models import BomPart

    bom = _nonsynchronous_buck_support()
    bom.no_connect_pins = []
    bom.parts.extend(
        BomPart(
            ref=ref, value=value, sheet="POWER", symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
        )
        for ref, value in (("R1", top), ("R2", bottom))
    )
    for connection in bom.connections:
        if connection.net_name == "VIN":
            connection.endpoints.append(SimpleNamespace(ref="R1", pin="1"))
        elif connection.net_name == "GND":
            connection.endpoints.append(SimpleNamespace(ref="R2", pin="2"))
    bom.connections.append(SimpleNamespace(
        net_name="ENABLE",
        endpoints=[
            SimpleNamespace(ref="U1", pin="3"),
            SimpleNamespace(ref="R1", pin="2"),
            SimpleNamespace(ref="R2", pin="1"),
        ],
    ))
    assert validation.check_reviewed_input_operating_ranges(
        SimpleNamespace(rail_voltages={"VIN": 5.0, "GND": 0.0}), bom,
    ).ok is accepted


# ---------- preregistered negative boundaries: I2C addresses, analog inputs ----------
#
# The held-out briefs are multi-ADS1115 boards, so the daq-8ch reference is the
# positive control for these gates: it must keep constructing and passing, and
# each mutated copy below must be refused with the diagnostic that names the
# address/input defect rather than a generic failure.

def _compose_daq8ch(mutate=None):
    """Construct the daq-8ch reference exactly as the architecture commit probe does."""
    from kicraft.design import models
    from kicraft.eval.design_acceptance import load_reference_rows
    from kicraft.server.stage_contracts import _normalize_stage_response
    from kicraft.server.stage_work_units import probe_architecture_construction

    row = next(
        row for _name, row in load_reference_rows()
        if (row.get("acceptance") or {}).get("slug") == "daq-8ch"
    )
    compiler_input = copy.deepcopy(row["compiler_input"])
    if mutate is not None:
        mutate(compiler_input)
    prompt_state = {
        "intent": compiler_input.get("intent") or {},
        "functional_spec": compiler_input.get("functional_spec") or {},
    }
    architecture = _normalize_stage_response(
        "architecture", compiler_input["architecture"], prompt_state,
    )
    if isinstance(architecture, tuple):
        architecture = architecture[0]
    built = probe_architecture_construction({**prompt_state, "architecture": architecture})
    assert built is not None
    return models.Architecture.model_validate(architecture), models.BOM.model_validate(built)


def test_multi_adc_reference_still_composes_and_passes_both_new_gates():
    architecture, bom = _compose_daq8ch()
    gates = {
        check.name: check for check in validation.check_composed_wiring(architecture, bom)
    }
    assert [name for name, check in gates.items() if not check.ok] == []
    assert gates["9.43 reviewed I2C address assignments"].ok
    assert gates["9.44 reviewed analog input ranges"].ok


def _duplicate_second_address_strap(compiler_input):
    """Strap both ADS1115 converters to GND, so both answer at 0x48."""
    for requirement in compiler_input["architecture"]["requirements"]:
        if requirement["id"] == "ads1115b":
            requirement["parameters"]["address_strap"] = "gnd"
            requirement["parameters"]["i2c_address"] = "0x48"


def _append_converters(straps):
    """Append one ADS1115 requirement per (suffix, address_strap, i2c_address)."""
    def mutate(compiler_input):
        requirements = compiler_input["architecture"]["requirements"]
        template = next(row for row in requirements if row["id"] == "ads1115a")
        for suffix, strap, address in straps:
            extra = copy.deepcopy(template)
            extra["id"] = f"ads1115{suffix}"
            extra["parameters"]["address_strap"] = strap
            extra["parameters"]["i2c_address"] = address
            requirements.append(extra)
    return mutate


_append_three_more_converters = _append_converters(
    [(suffix, "gnd", "0x48") for suffix in ("c", "d", "e")]
)


def _type_analog_input_voltage(volts):
    """Type one voltage on the converter's first single-ended input."""
    def mutate(compiler_input):
        compiler_input["architecture"]["rail_voltages"]["AI1"] = volts
    return mutate


_type_five_volts_on_the_first_analog_input = _type_analog_input_voltage(5.0)


def test_repeated_address_strap_on_one_bus_is_a_named_collision():
    architecture, bom = _compose_daq8ch(_duplicate_second_address_strap)
    result = validation.check_reviewed_i2c_address_assignments(architecture, bom)
    assert not result.ok
    assert any(
        offender.startswith("E_I2C_ADDRESS_COLLISION I2C_SDA: U1, U2 strap ADDR to 'GND'")
        and "declared 0x48" in offender
        and "reviewed 'ads1115idgsr'" in offender
        for offender in result.offenders
    )


def test_more_devices_than_address_straps_on_one_bus_is_named_capacity():
    architecture, bom = _compose_daq8ch(_append_three_more_converters)
    assert len([part for part in bom.parts if part.mpn == "ADS1115IDGSR"]) == 5
    result = validation.check_reviewed_i2c_address_assignments(architecture, bom)
    assert not result.ok
    assert any(
        offender.startswith(
            "E_I2C_ADDRESS_CAPACITY I2C_SDA: 5 reviewed 'ads1115idgsr' device(s) (U1, U2, U3, U4, U5)"
        )
        and "only 4 distinct address strap(s)" in offender
        for offender in result.offenders
    )


def test_direct_five_volt_input_on_a_three_volt_adc_is_a_named_overvoltage():
    architecture, bom = _compose_daq8ch(_type_five_volts_on_the_first_analog_input)
    result = validation.check_reviewed_analog_input_ranges(architecture, bom)
    assert not result.ok
    assert any(
        offender.startswith("E_ANALOG_INPUT_RANGE U1.4: AIN0 is wired to 'AI1', typed 5V")
        and "supply '+3V3' is 3.3V" in offender
        and "absolute maximum is 3.6V" in offender
        for offender in result.offenders
    )


@pytest.mark.parametrize(("input_volts", "accepted"), [(3.3, True), (3.6, True), (3.7, False)])
def test_analog_input_limit_is_the_device_supply_plus_its_reviewed_margin(input_volts, accepted):
    """VDD+0.3 V is the rated absolute maximum: at it the design is kept, above it refused."""
    architecture, bom = _compose_daq8ch(_type_analog_input_voltage(input_volts))
    assert validation.check_reviewed_analog_input_ranges(architecture, bom).ok is accepted


def test_four_converters_with_four_distinct_straps_are_not_refused():
    """The capacity gate counts straps, not devices: four straps address four converters."""
    architecture, bom = _compose_daq8ch(_append_converters(
        [("c", "sda", "0x4a"), ("d", "scl", "0x4b")],
    ))
    assert len([part for part in bom.parts if part.mpn == "ADS1115IDGSR"]) == 4
    assert validation.check_reviewed_i2c_address_assignments(architecture, bom).ok


@pytest.mark.parametrize(("mutate", "gate", "code"), [
    (_duplicate_second_address_strap, "9.43", "E_I2C_ADDRESS_COLLISION"),
    (_append_three_more_converters, "9.43", "E_I2C_ADDRESS_CAPACITY"),
    (_type_five_volts_on_the_first_analog_input, "9.44", "E_ANALOG_INPUT_RANGE"),
])
def test_architecture_commit_probe_refuses_each_mutated_payload(mutate, gate, code):
    """The commit probe reaches the gate and attributes it to the owning requirement."""
    from kicraft.design import models
    from kicraft.eval.design_acceptance import load_reference_rows
    from kicraft.server.stage_contracts import (
        StageSchemaError,
        _normalize_stage_response,
        _validate_constructed_architecture,
    )

    row = next(
        row for _name, row in load_reference_rows()
        if (row.get("acceptance") or {}).get("slug") == "daq-8ch"
    )
    compiler_input = copy.deepcopy(row["compiler_input"])
    mutate(compiler_input)
    prompt_state = {
        "intent": compiler_input.get("intent") or {},
        "functional_spec": compiler_input.get("functional_spec") or {},
    }
    architecture = _normalize_stage_response(
        "architecture", compiler_input["architecture"], prompt_state,
    )
    if isinstance(architecture, tuple):
        architecture = architecture[0]
    committed = models.Architecture.model_validate(architecture).model_dump(exclude_none=True)

    with pytest.raises(StageSchemaError) as raised:
        _validate_constructed_architecture(
            copy.deepcopy(committed),
            {**prompt_state, "architecture": copy.deepcopy(committed)},
            project_root=str(Path(__file__).resolve().parents[1]),
        )
    diagnostic = raised.value.diagnostic or {}
    assert diagnostic.get("code") == "constructed_circuit_invalid"
    findings = {finding["gate_codes"][0]: finding for finding in diagnostic["findings"]}
    assert gate in findings
    assert any(code in evidence for evidence in findings[gate]["evidence"])
    assert findings[gate]["candidate_requirement_ids"][gate]


def test_addressable_device_without_reviewed_address_data_is_named_not_refused(monkeypatch):
    """A record that declares no strap contract must not refuse, but must not be silent either."""
    monkeypatch.setattr(validation, "_reviewed_fact_for_part", lambda _part: {
        "mpn": "TCA9555PWR",
        "identity": "tca9555pwr",
        "physical_features": ["io-expander", "i2c-gpio-expander"],
        "port_pins": {"sda": "23", "scl": "22"},
    })
    monkeypatch.setattr(
        validation,
        "_pin_info_by_ref",
        lambda _bom: (
            {
                "U7": {
                    "23": {"name": "SDA", "type": "bidirectional"},
                    "22": {"name": "SCL", "type": "input"},
                }
            },
            {},
        ),
    )
    bom = _bom([_part("U7", "TCA9555PWR", mpn="TCA9555PWR")], {
        "I2C_SDA": [("U7", "23")],
        "I2C_SCL": [("U7", "22")],
    })
    result = validation.check_reviewed_i2c_address_assignments(
        SimpleNamespace(requirements=[]), bom,
    )
    assert result.ok
    assert "E_I2C_ADDRESS_UNREVIEWED U7" in result.message


def test_analog_input_device_without_reviewed_range_data_is_named_not_refused(monkeypatch):
    """An ADC record with no input-range contract is named, never refused on absent data."""
    monkeypatch.setattr(validation, "_reviewed_fact_for_part", lambda _part: {
        "mpn": "MCP3208",
        "identity": "mcp3208",
        "physical_features": ["adc"],
        "port_pins": {"vdd": "16", "ground": "15", "ain0": "1"},
    })
    monkeypatch.setattr(
        validation,
        "_pin_info_by_ref",
        lambda _bom: (
            {
                "U8": {
                    "1": {"name": "CH0", "type": "input"},
                    "16": {"name": "VDD", "type": "power_in"},
                }
            },
            {},
        ),
    )
    bom = _bom([_part("U8", "MCP3208", mpn="MCP3208")], {"CH0": [("U8", "1")]})
    result = validation.check_reviewed_analog_input_ranges(
        SimpleNamespace(rail_voltages={"CH0": 5.0}), bom,
    )
    assert result.ok
    assert "E_ANALOG_INPUT_RANGE U8" in result.message


def test_an_under_rated_selected_capacitor_is_repinned_before_the_gate(monkeypatch):
    """The capacitor §9.38 refuses is a recipe-owned passive, so the pin is repaired, not the model.

    Live cohort 2026-09-30: brief 3's wiring stage refused the same 10 V part on its 18 V rail four
    times ("E_CAPACITOR_VOLTAGE C1: selected C19702 is rated 10V but spans 18V across 'VIN18'/'GND'"),
    because the recipe's input capacitor has no voltage class, the tier-4 keyword pin chose the 10 V
    catalog row, and a recipe passive's pins are locked while the wiring response contract is
    pins-only -- so the refusal was unrepairable where it was raised. The selection is re-made here
    from the same offline catalog, keeping the value and package.
    """
    from kicraft.design import cli_app
    from kicraft.design.cli_app import _repin_under_rated_capacitors
    from kicraft.parts_library import jlcparts

    monkeypatch.setattr(
        jlcparts, "parameters",
        lambda cid: {"attributes": {"Voltage Rating": {"C19702": "10V", "C96446": "25V"}.get(cid, "")}},
    )
    monkeypatch.setattr(
        jlcparts, "search",
        lambda term, limit=10: [
            {
                "lcsc": "C19702", "model": "CL10A106KP8NNNC", "package": "0603",
                "description": "10uF ±10% 10V X5R 0603", "stock": 900000, "type": "Basic",
                "brand": "Samsung", "price": 0.01, "joints": 2,
            },
            {
                "lcsc": "C96446", "model": "CL10A106MA8NRNC", "package": "0603",
                "description": "10uF ±20% 25V X5R 0603", "stock": 400000, "type": "Basic",
                "brand": "Samsung", "price": 0.01, "joints": 2,
            },
        ],
    )
    monkeypatch.setattr(cli_app.lcsc_retail, "enabled", lambda: False)
    cap = SimpleNamespace(
        ref="C1", value="10uF", mpn=None, symbol="Test:Part", sheet="POWER",
        footprint="Capacitor_SMD:C_0603_1608Metric", sourcing_note="LCSC C19702",
    )
    bom = _bom([cap], {"VIN18": [("C1", "1")], "GND": [("C1", "2")]})
    bom.substitutions = []
    architecture = SimpleNamespace(rail_voltages={"VIN18": 18.0, "GND": 0.0})

    # The gate sees the under-rated selection first.
    assert not validation.check_reviewed_input_operating_ranges(architecture, bom).ok

    notes = _repin_under_rated_capacitors(architecture, bom)

    assert cap.sourcing_note == "LCSC C96446"
    assert any("re-pinned C19702 -> C96446" in note for note in notes)
    assert [(row.wanted, row.got) for row in bom.substitutions] == [
        ("C1 C19702 rated 10V", "C96446 rated 25V")
    ]
    # The repaired selection clears the gate that refused it.
    assert validation.check_reviewed_input_operating_ranges(architecture, bom).ok


def test_an_under_rated_capacitor_is_left_alone_when_no_rated_candidate_exists(monkeypatch):
    """Never invent a part: no rated catalog row means the gate still refuses, and says why."""
    from kicraft.design import cli_app
    from kicraft.design.cli_app import _repin_under_rated_capacitors
    from kicraft.parts_library import jlcparts

    monkeypatch.setattr(
        jlcparts, "parameters", lambda cid: {"attributes": {"Voltage Rating": "10V"}},
    )
    monkeypatch.setattr(
        jlcparts, "search",
        lambda term, limit=10: [
            {
                "lcsc": "C19702", "model": "CL10A106KP8NNNC", "package": "0603",
                "description": "10uF ±10% 10V X5R 0603", "stock": 900000, "type": "Basic",
                "brand": "Samsung", "price": 0.01, "joints": 2,
            },
        ],
    )
    monkeypatch.setattr(cli_app.lcsc_retail, "enabled", lambda: False)
    cap = SimpleNamespace(
        ref="C1", value="10uF", mpn=None, symbol="Test:Part", sheet="POWER",
        footprint="Capacitor_SMD:C_0603_1608Metric", sourcing_note="LCSC C19702",
    )
    bom = _bom([cap], {"VIN18": [("C1", "1")], "GND": [("C1", "2")]})
    bom.substitutions = []
    architecture = SimpleNamespace(rail_voltages={"VIN18": 18.0, "GND": 0.0})

    notes = _repin_under_rated_capacitors(architecture, bom)

    assert cap.sourcing_note == "LCSC C19702"
    assert not bom.substitutions
    assert any("no in-stock catalog part matching" in note for note in notes)
    assert not validation.check_reviewed_input_operating_ranges(architecture, bom).ok
