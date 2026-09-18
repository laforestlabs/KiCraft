"""Focused regressions for GAP1 device, transfer, and typed-value invariants."""
from __future__ import annotations

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
