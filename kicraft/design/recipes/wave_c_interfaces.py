"""Wave-C bus and driver recipes. Safety-sensitive drivers remain canary."""

from __future__ import annotations

from .models import (
    RecipeComponentGroup as Group,
    RecipeDefinition,
    RecipeElectricalAssertion as Assertion,
    RecipeNoConnectSpec as NoConnect,
    RecipePinSpec as Pin,
    RecipePlacementConstraint as Placement,
    RecipePort as Port,
    RecipeSourceDocument as Source,
)

_DATE = "2026-09-09"
_C = ("Device:C", "Capacitor_SMD:C_0603_1608Metric")
_R = ("Device:R", "Resistor_SMD:R_0603_1608Metric")


def _part(
    role: str,
    prefix: str,
    value: str,
    symbol: str,
    footprint: str,
    *,
    quantity: int = 1,
    mpn: str | None = None,
) -> Group:
    return Group(
        role=role,
        reference_prefix=prefix,
        value=value,
        symbol=symbol,
        footprint=footprint,
        sheet_role="interface",
        quantity=quantity,
        mpn=mpn,
    )


def _passive(role: str, prefix: str, value: str, quantity: int = 1) -> Group:
    symbol, footprint = _C if prefix == "C" else _R
    return _part(role, prefix, value, symbol, footprint, quantity=quantity)


def _recipe(
    *,
    recipe: str,
    family: str,
    exact: str,
    ports: tuple[Port, ...],
    internal: tuple[str, ...],
    parts: tuple[Group, ...],
    pins: tuple[Pin, ...],
    source: str | Source,
    assertions: tuple[Assertion, ...],
    no_connects: tuple[NoConnect, ...] = (),
    parameters: dict | None = None,
    allowed: dict | None = None,
    safety: bool = False,
) -> RecipeDefinition:
    constraints = [
        Placement(kind="decoupling_proximity", role=parts[0].role, parameters={"max_mm": 3.0})
    ]
    if safety:
        constraints.append(
            Placement(
                kind="thermal_current_path",
                role=parts[0].role,
                parameters={"independent_review_required": True},
            )
        )
    return RecipeDefinition(
        recipe=recipe,
        family=family,
        exact_part=exact,
        maturity="production",
        protected_aliases=(family, exact),
        identity_aliases=(exact,),
        required_sheet_roles=("interface",),
        parameter_defaults=parameters or {},
        allowed_parameters=allowed or {},
        ports=ports,
        internal_nets=internal,
        parts=parts,
        pins=pins,
        no_connects=no_connects,
        placement_constraints=tuple(constraints),
        electrical_assertions=assertions,
        source_documents=(
            source
            if isinstance(source, Source)
            else Source(
                url=source,
                title=f"{exact} data sheet",
                revision="current",
                reviewed_date=_DATE,
                sections=("Pin Functions", "Typical Application", "Layout Guidelines"),
            ),
        ),
    )


SN65HVD230_CAN_NODE = _recipe(
    recipe="sn65hvd230-can-node@1",
    family="can-node-3v3",
    exact="SN65HVD230",
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="tx", direction="input"),
        Port(name="rx", direction="output"),
        Port(name="canh", direction="bidirectional"),
        Port(name="canl", direction="bidirectional"),
        Port(name="stby", direction="input", required=False),
    ),
    internal=("can_term",),
    parts=(
        _part(
            "transceiver",
            "U",
            "SN65HVD230",
            "Interface_CAN_LIN:SN65HVD230",
            "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
            mpn="SN65HVD230",
        ),
        _passive("decoupling", "C", "100nF"),
        _passive("slope_resistor", "R", "10k"),
        _passive("termination", "R", "120R"),
        _part(
            "termination_jumper",
            "JP",
            "TERM",
            "Jumper:SolderJumper_2_Open",
            "Jumper:SolderJumper-2_P1.3mm_Open_RoundedPad1.0x1.5mm",
        ),
    ),
    pins=(
        Pin(role="transceiver", pin="1", net="tx"),
        Pin(role="transceiver", pin="4", net="rx"),
        Pin(role="transceiver", pin="3", net="vdd"),
        Pin(role="transceiver", pin="2", net="gnd"),
        Pin(role="transceiver", pin="7", net="canh"),
        Pin(role="transceiver", pin="6", net="canl"),
        Pin(role="transceiver", pin="8", net="stby"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
        Pin(role="slope_resistor", pin="1", net="stby"),
        Pin(role="slope_resistor", pin="2", net="gnd"),
        Pin(role="termination", pin="1", net="canh"),
        Pin(role="termination", pin="2", net="can_term"),
        Pin(role="termination_jumper", pin="1", net="can_term"),
        Pin(role="termination_jumper", pin="2", net="canl"),
    ),
    no_connects=(
        NoConnect(
            role="transceiver",
            pin="5",
            # Unused VCC/2 reference output, not a ground pin.
        ),
    ),
    source="https://www.ti.com/lit/ds/symlink/sn65hvd230.pdf",
    assertions=(
        Assertion(
            code="can_switchable_termination",
            message="120 Ohm termination is installed only by the explicit solder jumper",
        ),
    ),
)

MAX3485_RS485_NODE = _recipe(
    recipe="max3485-rs485-node@1",
    family="rs485-node-3v3",
    exact="MAX3485",
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="tx", direction="input"),
        Port(name="rx", direction="output"),
        Port(name="de", direction="input"),
        Port(name="re_n", direction="input"),
        Port(name="a", direction="bidirectional"),
        Port(name="b", direction="bidirectional"),
    ),
    internal=(),
    parts=(
        _part(
            "transceiver",
            "U",
            "MAX3485",
            "Interface_UART:MAX3485",
            "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
            mpn="MAX3485",
        ),
        _passive("decoupling", "C", "100nF"),
        _passive("termination", "R", "120R"),
    ),
    pins=(
        Pin(role="transceiver", pin="1", net="rx"),
        Pin(role="transceiver", pin="2", net="re_n"),
        Pin(role="transceiver", pin="3", net="de"),
        Pin(role="transceiver", pin="4", net="tx"),
        Pin(role="transceiver", pin="8", net="vdd"),
        Pin(role="transceiver", pin="5", net="gnd"),
        Pin(role="transceiver", pin="6", net="a"),
        Pin(role="transceiver", pin="7", net="b"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
        Pin(role="termination", pin="1", net="a"),
        Pin(role="termination", pin="2", net="b"),
    ),
    source="https://www.analog.com/media/en/technical-documentation/data-sheets/MAX3483-MAX3491.pdf",
    assertions=(
        Assertion(
            code="rs485_termination_policy",
            message="Termination is owned by this node and must only be populated at a bus endpoint",
        ),
    ),
)

DRV8833_DUAL_MOTOR = _recipe(
    recipe="drv8833-dual-motor@1",
    family="dual-dc-motor-driver",
    exact="DRV8833PWPR",
    safety=True,
    ports=(
        Port(name="vm", direction="power"),
        Port(name="gnd", direction="power"),
        *(
            Port(name=name, direction="input")
            for name in ("ain1", "ain2", "bin1", "bin2", "sleep_n")
        ),
        Port(name="fault_n", direction="output"),
        *(Port(name=name, direction="output") for name in ("aout1", "aout2", "bout1", "bout2")),
    ),
    internal=("vcp", "vint", "aisense", "bisense"),
    parts=(
        _part(
            "driver",
            "U",
            "DRV8833PWPR",
            "drv8833:DRV8833PWP",
            "drv8833:TSSOP-16_L5.0-W4.4-P0.65-LS6.4-BL-EP",
            mpn="DRV8833PWPR",
        ),
        _passive("vm_bulk", "C", "47uF"),
        _passive("vm_decoupling", "C", "100nF"),
        _passive("vcp_cap", "C", "10nF"),
        _passive("vint_cap", "C", "2.2uF"),
        _passive("sense", "R", "0R", 2),
    ),
    pins=(
        Pin(role="driver", pin="1", net="sleep_n"),
        Pin(role="driver", pin="2", net="aout1"),
        Pin(role="driver", pin="3", net="aisense"),
        Pin(role="driver", pin="4", net="aout2"),
        Pin(role="driver", pin="5", net="bout2"),
        Pin(role="driver", pin="6", net="bisense"),
        Pin(role="driver", pin="7", net="bout1"),
        Pin(role="driver", pin="8", net="fault_n"),
        Pin(role="driver", pin="9", net="bin1"),
        Pin(role="driver", pin="10", net="bin2"),
        Pin(role="driver", pin="11", net="vcp"),
        Pin(role="driver", pin="12", net="vm"),
        Pin(role="driver", pin="13", net="gnd"),
        Pin(role="driver", pin="14", net="vint"),
        Pin(role="driver", pin="15", net="ain2"),
        Pin(role="driver", pin="16", net="ain1"),
        Pin(role="driver", pin="17", net="gnd"),
        *(
            Pin(role=role, pin="1", net=net)
            for role, net in (
                ("vm_bulk", "vm"),
                ("vm_decoupling", "vm"),
                ("vcp_cap", "vcp"),
                ("vint_cap", "vint"),
            )
        ),
        *(
            Pin(role=role, pin="2", net="gnd")
            for role in ("vm_bulk", "vm_decoupling", "vcp_cap", "vint_cap")
        ),
        Pin(role="sense", index=0, pin="1", net="aisense"),
        Pin(role="sense", index=0, pin="2", net="gnd"),
        Pin(role="sense", index=1, pin="1", net="bisense"),
        Pin(role="sense", index=1, pin="2", net="gnd"),
    ),
    source="https://www.ti.com/lit/ds/symlink/drv8833.pdf",
    assertions=(
        Assertion(
            code="motor_current_thermal_bound",
            message="Motor stall current, sense configuration, copper area, and bulk capacitance require independent review",
        ),
    ),
)

A4988_STEPPER = _recipe(
    recipe="a4988-stepper@1",
    family="a4988-stepper-driver",
    exact="A4988SETTR-T",
    safety=True,
    ports=(
        Port(name="vm", direction="power"),
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="step", direction="input"),
        Port(name="dir", direction="input"),
        *(Port(name=name, direction="output") for name in ("out1a", "out1b", "out2a", "out2b")),
    ),
    internal=("cp1", "cp2", "vcp", "vreg", "sense1", "sense2", "ref", "reset_sleep"),
    parts=(
        _part(
            "driver",
            "U",
            "A4988SETTR-T",
            "a4988:A4988SETTR-T",
            "a4988:WQFN-28_L5.0-W5.0-P0.50-BL-EP3.2",
            mpn="A4988SETTR-T",
        ),
        _passive("charge_pump", "C", "100nF", 2),
        _passive("vreg_cap", "C", "1uF"),
        _passive("vm_bulk", "C", "100uF"),
        _passive("sense", "R", "0.1R", 2),
        _passive("ref_top", "R", "33k"),
        _passive("ref_bottom", "R", "10k"),
        _passive("reset_pullup", "R", "10k"),
    ),
    pins=(
        Pin(role="driver", pin="1", net="out2b"),
        Pin(role="driver", pin="2", net="gnd"),
        Pin(role="driver", pin="3", net="gnd"),
        Pin(role="driver", pin="4", net="cp1"),
        Pin(role="driver", pin="5", net="cp2"),
        Pin(role="driver", pin="6", net="vcp"),
        Pin(role="driver", pin="8", net="vreg"),
        Pin(role="driver", pin="9", net="gnd"),
        Pin(role="driver", pin="10", net="gnd"),
        Pin(role="driver", pin="11", net="gnd"),
        Pin(role="driver", pin="12", net="reset_sleep"),
        Pin(role="driver", pin="13", net="gnd"),
        Pin(role="driver", pin="14", net="reset_sleep"),
        Pin(role="driver", pin="15", net="vdd"),
        Pin(role="driver", pin="16", net="step"),
        Pin(role="driver", pin="17", net="ref"),
        Pin(role="driver", pin="18", net="gnd"),
        Pin(role="driver", pin="19", net="dir"),
        Pin(role="driver", pin="21", net="out1b"),
        Pin(role="driver", pin="22", net="vm"),
        Pin(role="driver", pin="23", net="sense1"),
        Pin(role="driver", pin="24", net="out1a"),
        Pin(role="driver", pin="26", net="out2a"),
        Pin(role="driver", pin="27", net="sense2"),
        Pin(role="driver", pin="28", net="vm"),
        Pin(role="driver", pin="29", net="gnd"),
        Pin(role="charge_pump", index=0, pin="1", net="cp1"),
        Pin(role="charge_pump", index=0, pin="2", net="cp2"),
        Pin(role="charge_pump", index=1, pin="1", net="vcp"),
        Pin(role="charge_pump", index=1, pin="2", net="vm"),
        Pin(role="vreg_cap", pin="1", net="vreg"),
        Pin(role="vreg_cap", pin="2", net="gnd"),
        Pin(role="vm_bulk", pin="1", net="vm"),
        Pin(role="vm_bulk", pin="2", net="gnd"),
        *(Pin(role="sense", index=i, pin="1", net=f"sense{i + 1}") for i in range(2)),
        *(Pin(role="sense", index=i, pin="2", net="gnd") for i in range(2)),
        Pin(role="ref_top", pin="1", net="vdd"),
        Pin(role="ref_top", pin="2", net="ref"),
        Pin(role="ref_bottom", pin="1", net="ref"),
        Pin(role="ref_bottom", pin="2", net="gnd"),
        Pin(role="reset_pullup", pin="1", net="vdd"),
        Pin(role="reset_pullup", pin="2", net="reset_sleep"),
    ),
    no_connects=tuple(NoConnect(role="driver", pin=pin) for pin in ("7", "20", "25")),
    source="https://www.allegromicro.com/-/media/files/datasheets/a4988-datasheet.pdf",
    assertions=(
        Assertion(
            code="stepper_current_limit",
            message="33 kOhm/10 kOhm reference divider with 0.1 Ohm sense resistors sets about 0.96 A full-scale; motor and thermal limits require review",
        ),
    ),
)


def _i2c_device(
    *,
    recipe: str,
    family: str,
    exact: str,
    symbol: str,
    footprint: str,
    datasheet: str,
    power_pin: str,
    ground_pin: str,
    sda_pin: str,
    scl_pin: str,
    io_pins: tuple[str, ...],
    address_pins: tuple[str, ...],
    interrupt_pin: str | None = None,
) -> RecipeDefinition:
    ports = [
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="sda", direction="bidirectional"),
        Port(name="scl", direction="input"),
    ]
    pins = [
        Pin(role="device", pin=power_pin, net="vdd"),
        Pin(role="device", pin=ground_pin, net="gnd"),
        Pin(role="device", pin=sda_pin, net="sda"),
        Pin(role="device", pin=scl_pin, net="scl"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
    ]
    for i, pin in enumerate(io_pins):
        ports.append(Port(name=f"io{i}", direction="bidirectional"))
        pins.append(Pin(role="device", pin=pin, net=f"io{i}"))
    for pin in address_pins:
        pins.append(Pin(role="device", pin=pin, net="gnd"))
    if interrupt_pin:
        ports.append(Port(name="interrupt", direction="output", required=False))
        pins.append(Pin(role="device", pin=interrupt_pin, net="interrupt"))
    return _recipe(
        recipe=recipe,
        family=family,
        exact=exact,
        ports=tuple(ports),
        internal=(),
        parts=(
            _part("device", "U", exact, symbol, footprint, mpn=exact),
            _passive("decoupling", "C", "100nF"),
        ),
        pins=tuple(pins),
        source=datasheet,
        assertions=(
            Assertion(
                code="i2c_address_owned",
                message="Address straps and pull-up ownership are explicit",
            ),
        ),
    )


PCA9685_SERVO_BANK = _i2c_device(
    recipe="pca9685-servo-bank@1",
    family="pca9685-servo-bank",
    exact="PCA9685PW,118",
    symbol="pca9685:PCA9685PW118",
    footprint="pca9685:TSSOP-28_L9.7-W4.4-P0.65-LS6.4-TL",
    datasheet="https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf",
    power_pin="28",
    ground_pin="14",
    sda_pin="27",
    scl_pin="26",
    io_pins=(
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
    ),
    address_pins=("1", "2", "3", "4", "5", "24", "23", "25"),
)
TCA9555_GPIO_EXPANDER = _i2c_device(
    recipe="tca9555-gpio-expander@1",
    family="tca9555-gpio-expander",
    exact="TCA9555PWR(UMW)",
    symbol="tca9555:TCA9555PWR",
    footprint="tca9555:TSSOP-24_L7.8-W4.4-P0.65-LS6.4-BL",
    datasheet="https://www.ti.com/lit/ds/symlink/tca9555.pdf",
    power_pin="24",
    ground_pin="12",
    sda_pin="23",
    scl_pin="22",
    io_pins=(
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
    ),
    address_pins=("21", "2", "3"),
    interrupt_pin="1",
)
ADS1115_I2C_ADC = _i2c_device(
    recipe="ads1115-i2c-adc@1",
    family="ads1115-i2c-adc",
    exact="ADS1115IDGSR",
    symbol="ads1115:ADS1115IDGSR",
    footprint="ads1115:MSOP-10_L3.0-W3.0-P0.50-LS5.0-BL",
    datasheet="https://www.ti.com/lit/ds/symlink/ads1115.pdf",
    power_pin="8",
    ground_pin="3",
    sda_pin="9",
    scl_pin="10",
    io_pins=("4", "5", "6", "7"),
    address_pins=("1",),
    interrupt_pin="2",
)

# HUB75 (16-pin IDC) logic channels, in connector-pin order. Ground is pins 4, 8
# and 16: pin 12 is the D address line (used by 1/32-scan panels), NOT a ground —
# grounding it silently dropped the D channel. The two '245 banks carry all
# thirteen channels, so U1 uses five of its eight and its three spare channels
# are no-connects.
_HUB75_SIGNALS = (
    ("r0", 1), ("g0", 2), ("b0", 3),
    ("r1", 5), ("g1", 6), ("b1", 7),
    ("addr_a", 9), ("addr_b", 10), ("addr_c", 11), ("addr_d", 12),
    ("clk", 13), ("lat", 14), ("oe", 15),
)
_HUB75_GROUND_PINS = (4, 8, 16)
HUB75_SN74HCT245_INTERFACE = _recipe(
    recipe="hub75-sn74hct245-interface@1",
    family="hub75-level-shift-interface",
    exact="HUB75-SN74HCT245",
    ports=(
        Port(name="vdd_5v", direction="power"),
        Port(name="gnd", direction="power"),
        # Semantic channel names, so a model can bind the interface it declared
        # (HUB75_OE, HUB75_CLK, …) by name instead of guessing input0..input11.
        *(Port(name=name, direction="input") for name, _pin in _HUB75_SIGNALS),
    ),
    internal=tuple(f"shifted{i}" for i in range(len(_HUB75_SIGNALS))),

    parts=(
        _part(
            "level_shifter",
            "U",
            "SN74HCT245PWR-JSM",
            "sn74hct245:SN74HCT245PWR-JSM",
            "sn74hct245:TSSOP-20_L6.5-W4.4-P0.65-LS6.4-BL",
            quantity=2,
            mpn="SN74HCT245PWR-JSM",
        ),
        _part(
            "connector",
            "J",
            "HUB75",
            "Connector_Generic:Conn_02x08_Odd_Even",
            "Connector_IDC:IDC-Header_2x08_P2.54mm_Vertical",
        ),
        _passive("decoupling", "C", "100nF", 2),
    ),
    pins=(
        *(Pin(role="level_shifter", index=u, pin="20", net="vdd_5v") for u in range(2)),
        *(Pin(role="level_shifter", index=u, pin="10", net="gnd") for u in range(2)),
        *(Pin(role="level_shifter", index=u, pin="1", net="vdd_5v") for u in range(2)),
        *(Pin(role="level_shifter", index=u, pin="19", net="gnd") for u in range(2)),
        *(Pin(role="decoupling", index=u, pin="1", net="vdd_5v") for u in range(2)),
        *(Pin(role="decoupling", index=u, pin="2", net="gnd") for u in range(2)),
        *(
            pin
            for i, (name, connector_pin) in enumerate(_HUB75_SIGNALS)
            for pin in (
                Pin(role="level_shifter", index=i // 8, pin=str(2 + i % 8), net=name),
                Pin(role="level_shifter", index=i // 8, pin=str(18 - i % 8), net=f"shifted{i}"),
                Pin(role="connector", pin=str(connector_pin), net=f"shifted{i}"),
            )
        ),
        *(Pin(role="connector", pin=str(pin), net="gnd") for pin in _HUB75_GROUND_PINS),
    ),
    no_connects=tuple(
        NoConnect(role="level_shifter", index=1, pin=str(pin))
        for pin in (7, 8, 9, 11, 12, 13)
    ),
    source="https://www.ti.com/lit/ds/symlink/sn74hct245.pdf",
    assertions=(
        Assertion(
            code="hub75_level_translation",
            message=(
                "All thirteen HUB75 logic channels (R0 G0 B0 R1 G1 B1 A B C D CLK "
                "LAT OE) pass through HCT-family 3.3 V to 5 V translation"
            ),
        ),
    ),
)

WS2812_OUTPUT = _recipe(
    recipe="ws2812-output@1",
    family="ws2812-output",
    exact="WS2812B-B/T",
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="data_in", direction="input"),
        Port(name="data_out", direction="output"),
    ),
    internal=("data_series",),
    parts=(
        _part(
            "led",
            "D",
            "WS2812B-B/T",
            "ws2812b:WS2812B-B_W",
            "ws2812b:LED-SMD_4P-L5.0-W5.0-TL_WS2812B-B",
            mpn="WS2812B-B/T",
        ),
        _passive("series_resistor", "R", "330R"),
        _passive("decoupling", "C", "100nF"),
        _passive("bulk", "C", "10uF"),
    ),
    pins=(
        Pin(role="led", pin="1", net="vdd"),
        Pin(role="led", pin="2", net="data_out"),
        Pin(role="led", pin="3", net="gnd"),
        Pin(role="led", pin="4", net="data_series"),
        Pin(role="series_resistor", pin="1", net="data_in"),
        Pin(role="series_resistor", pin="2", net="data_series"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
        Pin(role="bulk", pin="1", net="vdd"),
        Pin(role="bulk", pin="2", net="gnd"),
    ),
    source="https://www.world-semi.com/ws2812-family/239.html",
    assertions=(
        Assertion(
            code="ws2812_logic_level",
            message="Controller high level must meet the LED VIH bound at the selected supply",
        ),
    ),
)

CH340C_USB_UART = _recipe(
    recipe="ch340c-usb-uart@1",
    family="usb-uart-bridge",
    exact="CH340C",
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="usb_dp", direction="bidirectional"),
        Port(name="usb_dm", direction="bidirectional"),
        Port(name="tx", direction="output"),
        Port(name="rx", direction="input"),
        Port(name="dtr_n", direction="output", required=False),
        Port(name="rts_n", direction="output", required=False),
    ),
    parameters={"supply_voltage": 3.3},
    allowed={"supply_voltage": (3.3, 5.0)},
    internal=(),
    parts=(
        _part(
            "bridge",
            "U",
            "CH340C",
            "ch340c:CH340C",
            "ch340c:SOP-16_L10.0-W3.9-P1.27-LS6.0-BL",
            mpn="CH340C",
        ),
        _passive("decoupling", "C", "100nF"),
        _passive("v3_cap", "C", "100nF"),
    ),
    pins=(
        Pin(role="bridge", pin="1", net="gnd"),
        Pin(role="bridge", pin="2", net="tx"),
        Pin(role="bridge", pin="3", net="rx"),
        Pin(role="bridge", pin="4", net="vdd"),
        Pin(role="bridge", pin="5", net="usb_dp"),
        Pin(role="bridge", pin="6", net="usb_dm"),
        Pin(role="bridge", pin="13", net="dtr_n"),
        Pin(role="bridge", pin="14", net="rts_n"),
        Pin(role="bridge", pin="16", net="vdd"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
        Pin(role="v3_cap", pin="1", net="vdd"),
        Pin(role="v3_cap", pin="2", net="gnd"),
    ),
    no_connects=tuple(
        NoConnect(role="bridge", pin=pin) for pin in ("7", "8", "9", "10", "11", "12", "15")
    ),
    source=Source(
        url="https://www.wch-ic.com/downloads/CH340DS1_PDF.html",
        title="CH340 data sheet",
        revision="3D",
        reviewed_date="2026-09-11",
        sections=(
            "5.1 Clock, Reset, Power, Connection",
            "6.2 5V Electrical Parameters",
            "6.3 3.3V Electrical Parameters",
            "7.6 Connect to MCU Serial Port, Unified Power Supply",
            "7.7 Connect to MCU, Supply Power to Each, and Prevent Flooding in Both Directions",
        ),
    ),
    assertions=(
        Assertion(
            code="usb_uart_power_mode",
            message=(
                "CH340DS1 v3D §5.1: at 3.3 V, V3 and VCC share vdd; at 5 V, "
                "V3 is a private node with only a 100 nF bypass to ground. "
                "UART/control levels depend on VCC (§6.2); separate MCU supplies "
                "require reviewed isolation/translation (§§7.6–7.7)."
            ),
        ),
    ),
)


def expand_ch340c_usb_uart(resolved):
    """Select the manufacturer's V3 topology without changing external rails."""
    from .registry import expand_static_definition

    definition = CH340C_USB_UART
    supply_voltage = resolved.parameters["supply_voltage"]
    if supply_voltage == 5.0:
        definition = definition.model_copy(
            update={
                "internal_nets": ("v3",),
                "pins": tuple(
                    pin.model_copy(update={"net": "v3"})
                    if (pin.role, pin.pin) in {("bridge", "4"), ("v3_cap", "1")}
                    else pin
                    for pin in definition.pins
                ),
            }
        )
    elif supply_voltage != 3.3:
        raise ValueError(f"recipe {definition.recipe} supply_voltage must be 3.3 or 5.0")
    return expand_static_definition(definition, resolved)


WAVE_C_INTERFACE_RECIPES = (
    SN65HVD230_CAN_NODE,
    MAX3485_RS485_NODE,
    DRV8833_DUAL_MOTOR,
    A4988_STEPPER,
    PCA9685_SERVO_BANK,
    TCA9555_GPIO_EXPANDER,
    ADS1115_I2C_ADC,
    HUB75_SN74HCT245_INTERFACE,
    WS2812_OUTPUT,
    CH340C_USB_UART,
)
