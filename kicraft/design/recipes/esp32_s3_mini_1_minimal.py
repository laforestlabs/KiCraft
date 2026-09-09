"""Reviewed ESP32-S3-MINI-1-N8 module core recipe.

The module pin table is from Espressif ESP32-S3-MINI-1/MINI-1U Datasheet v1.7,
Table 3-1. Power, CHIP_PU RC, GPIO0 boot, and USB series networks follow the
Espressif ESP32-S3 Hardware Design Guidelines schematic checklist.
"""

from __future__ import annotations

from .models import (
    RecipeAllocatablePin as AllocatablePin,
    RecipeComponentGroup as Group,
    RecipeDefinition,
    RecipeElectricalAssertion as Assertion,
    RecipeNoConnectSpec as NoConnect,
    RecipePinSpec as Pin,
    RecipePlacementConstraint as PlacementConstraint,
    RecipePort as Port,
    RecipeSourceDocument as SourceDocument,
    ResolvedRecipeSelection,
)
from .registry import expand_static_definition

_RECIPE_ID = "esp32-s3-mini-1-minimal@1"
_MODULE_DATASHEET = (
    "https://www.espressif.com/sites/default/files/documentation/"
    "esp32-s3-mini-1_mini-1u_datasheet_en.pdf"
)
_GUIDELINES = (
    "https://docs.espressif.com/projects/esp-hardware-design-guidelines/"
    "en/latest/esp32s3/schematic-checklist.html"
)

_GPIO_TO_PIN = {
    0: "4",
    1: "5",
    2: "6",
    3: "7",
    **{gpio: str(gpio + 4) for gpio in range(4, 22)},
    26: "26",
    47: "27",
    33: "28",
    34: "29",
    48: "30",
    35: "31",
    36: "32",
    37: "33",
    38: "34",
    39: "35",
    40: "36",
    41: "37",
    42: "38",
    43: "39",
    44: "40",
    45: "41",
    46: "44",
}
_STRAPPING_GPIOS = {0, 3, 45, 46}
_FIXED_GPIO = {0, 3, 19, 20, 43, 44, 45, 46}
_GROUND_PINS = (
    "1",
    "2",
    "42",
    "43",
    *(str(number) for number in range(46, 61)),
    "GND",
)


def _capabilities(gpio: int) -> tuple[str, ...]:
    values = [
        "gpio",
        "input",
        "output",
        "pwm",
        "i2c-sda",
        "i2c-scl",
        "spi-sclk",
        "spi-mosi",
        "spi-miso",
        "spi-cs",
        "uart-tx",
        "uart-rx",
        "interrupt",
    ]
    if 1 <= gpio <= 20:
        values.append("adc")
    if 1 <= gpio <= 14:
        values.append("touch")
    return tuple(values)


ESP32_S3_MINI_1_MINIMAL = RecipeDefinition(
    recipe=_RECIPE_ID,
    family="esp32-s3-module",
    exact_part="ESP32-S3-MINI-1-N8",
    default_for_family=True,
    maturity="production",
    identity_aliases=(
        "ESP32-S3-MINI-1",
        "ESP32-S3-MINI-1-N8",
        "esp32-s3-mini-1:ESP32-S3-MINI-1-N8",
    ),
    protected_aliases=("ESP32-S3", "ESP32-S3 module"),
    required_sheet_roles=("mcu",),
    parameter_defaults={"native_usb": False},
    allowed_parameters={"native_usb": (False, True)},
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="usb_dm", direction="bidirectional", required=False),
        Port(name="usb_dp", direction="bidirectional", required=False),
    ),
    internal_nets=(
        "en_rc",
        "boot_gpio0",
        "uart0_tx",
        "uart0_rx",
        "usb_dm_raw",
        "usb_dp_raw",
    ),
    parts=(
        Group(
            role="mcu",
            reference_prefix="U",
            value="ESP32-S3-MINI-1-N8",
            symbol="esp32-s3-mini-1:ESP32-S3-MINI-1-N8",
            footprint="esp32-s3-mini-1:BULETM-SMD_ESP32-S3-MINI-1-N8",
            sheet_role="mcu",
            mpn="ESP32-S3-MINI-1-N8",
            datasheet=_MODULE_DATASHEET,
        ),
        Group(
            role="bulk_decoupling",
            reference_prefix="C",
            value="10uF",
            symbol="Device:C",
            footprint="Capacitor_SMD:C_0603_1608Metric",
            sheet_role="mcu",
        ),
        Group(
            role="local_decoupling",
            reference_prefix="C",
            value="100nF",
            symbol="Device:C",
            footprint="Capacitor_SMD:C_0603_1608Metric",
            sheet_role="mcu",
        ),
        Group(
            role="en_capacitor",
            reference_prefix="C",
            value="1uF",
            symbol="Device:C",
            footprint="Capacitor_SMD:C_0603_1608Metric",
            sheet_role="mcu",
        ),
        Group(
            role="en_pullup",
            reference_prefix="R",
            value="10k",
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
            sheet_role="mcu",
        ),
        Group(
            role="boot_pullup",
            reference_prefix="R",
            value="10k",
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
            sheet_role="mcu",
        ),
        Group(
            role="reset_button",
            reference_prefix="SW",
            value="RESET",
            symbol="Switch:SW_Push",
            footprint="Button_Switch_SMD:SW_SPST_TL3342",
            sheet_role="mcu",
        ),
        Group(
            role="boot_button",
            reference_prefix="SW",
            value="BOOT",
            symbol="Switch:SW_Push",
            footprint="Button_Switch_SMD:SW_SPST_TL3342",
            sheet_role="mcu",
        ),
        Group(
            role="program_header",
            reference_prefix="J",
            value="ESP32-S3 UART PROGRAM",
            symbol="Connector_Generic:Conn_01x06",
            footprint=(
                "Connector_PinHeader_2.54mm:"
                "PinHeader_1x06_P2.54mm_Vertical"
            ),
            sheet_role="mcu",
        ),
    ),
    pins=(
        Pin(role="mcu", pin="3", net="vdd"),
        *(Pin(role="mcu", pin=pin, net="gnd") for pin in _GROUND_PINS),
        Pin(role="mcu", pin="45", net="en_rc"),
        Pin(role="mcu", pin="4", net="boot_gpio0"),
        Pin(role="bulk_decoupling", pin="1", net="vdd"),
        Pin(role="bulk_decoupling", pin="2", net="gnd"),
        Pin(role="local_decoupling", pin="1", net="vdd"),
        Pin(role="local_decoupling", pin="2", net="gnd"),
        Pin(role="en_capacitor", pin="1", net="en_rc"),
        Pin(role="en_capacitor", pin="2", net="gnd"),
        Pin(role="en_pullup", pin="1", net="vdd"),
        Pin(role="en_pullup", pin="2", net="en_rc"),
        Pin(role="boot_pullup", pin="1", net="vdd"),
        Pin(role="boot_pullup", pin="2", net="boot_gpio0"),
        Pin(role="reset_button", pin="1", net="en_rc"),
        Pin(role="reset_button", pin="2", net="gnd"),
        Pin(role="boot_button", pin="1", net="boot_gpio0"),
        Pin(role="boot_button", pin="2", net="gnd"),
        Pin(role="mcu", pin="39", net="uart0_tx"),
        Pin(role="mcu", pin="40", net="uart0_rx"),
        Pin(role="program_header", pin="1", net="vdd"),
        Pin(role="program_header", pin="2", net="gnd"),
        Pin(role="program_header", pin="3", net="uart0_tx"),
        Pin(role="program_header", pin="4", net="uart0_rx"),
        Pin(role="program_header", pin="5", net="en_rc"),
        Pin(role="program_header", pin="6", net="boot_gpio0"),
    ),
    no_connects=(
        NoConnect(role="mcu", pin="7"),
        NoConnect(role="mcu", pin="41"),
        NoConnect(role="mcu", pin="44"),
    ),
    allocatable_pins=tuple(
        AllocatablePin(
            role="mcu",
            pin=pin,
            gpio=gpio,
            capabilities=_capabilities(gpio),
            strapping=gpio in _STRAPPING_GPIOS,
            reserved=gpio in _FIXED_GPIO,
        )
        for gpio, pin in sorted(_GPIO_TO_PIN.items())
        if gpio not in _FIXED_GPIO
    ),
    placement_constraints=(
        PlacementConstraint(
            kind="antenna_keepout",
            role="mcu",
            parameters={"antenna_edge": "module_top", "copper_keepout": True},
        ),
        PlacementConstraint(
            kind="decoupling_proximity",
            role="local_decoupling",
            parameters={"anchor_role": "mcu", "max_mm": 3.0},
        ),
        PlacementConstraint(
            kind="usb_differential_pair",
            role="mcu",
            parameters={"impedance_ohm": 90.0, "max_skew_mm": 0.5},
        ),
    ),
    electrical_assertions=(
        Assertion(
            code="esp32_s3_supply_3v3",
            message="VDD must be 3.0 V through 3.6 V with at least 500 mA available",
        ),
        Assertion(
            code="esp32_s3_en_rc",
            message="EN is held by the reviewed 10 kOhm / 1 uF reset network",
        ),
        Assertion(
            code="esp32_s3_boot_access",
            message="GPIO0 and EN have physical BOOT and RESET access",
        ),
        Assertion(
            code="esp32_s3_programming_access",
            message="UART0, EN, GPIO0, VDD, and GND reach the programming header",
        ),
    ),
    source_documents=(
        SourceDocument(
            url=_MODULE_DATASHEET,
            title="ESP32-S3-MINI-1 & MINI-1U Datasheet",
            revision="v1.7",
            reviewed_date="2026-09-09",
            sections=("3 Pin Definitions", "4 Boot Configurations", "11 PCB Layout Recommendations"),
        ),
        SourceDocument(
            url=_GUIDELINES,
            title="ESP32-S3 Hardware Design Guidelines: Schematic Checklist",
            revision="latest reviewed 2026-09-09",
            reviewed_date="2026-09-09",
            sections=("Power Supply", "Chip Power-up and Reset Timing", "Strapping Pins", "USB"),
        ),
    ),
)


def expand_esp32_s3_mini_1(
    resolved: ResolvedRecipeSelection,
):
    """Materialize optional USB and no-connect every unallocated application GPIO."""
    native_usb = bool(resolved.parameters["native_usb"])
    selection = resolved.selection
    if native_usb and not {"usb_dm", "usb_dp"} <= set(selection.port_bindings):
        raise ValueError(
            f"recipe {_RECIPE_ID} native_usb=True requires usb_dm and usb_dp bindings"
        )
    parts = list(ESP32_S3_MINI_1_MINIMAL.parts)
    pins = list(ESP32_S3_MINI_1_MINIMAL.pins)
    no_connects = list(ESP32_S3_MINI_1_MINIMAL.no_connects)
    if native_usb:
        parts.extend(
            [
                Group(
                    role="usb_dm_series",
                    reference_prefix="R",
                    value="22R",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    sheet_role="mcu",
                ),
                Group(
                    role="usb_dp_series",
                    reference_prefix="R",
                    value="22R",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    sheet_role="mcu",
                ),
            ]
        )
        pins.extend(
            [
                Pin(role="mcu", pin="23", net="usb_dm_raw"),
                Pin(role="usb_dm_series", pin="1", net="usb_dm_raw"),
                Pin(role="usb_dm_series", pin="2", net="usb_dm"),
                Pin(role="mcu", pin="24", net="usb_dp_raw"),
                Pin(role="usb_dp_series", pin="1", net="usb_dp_raw"),
                Pin(role="usb_dp_series", pin="2", net="usb_dp"),
            ]
        )
    else:
        no_connects.extend(
            [NoConnect(role="mcu", pin="23"), NoConnect(role="mcu", pin="24")]
        )
    allocated = {allocation.pin for allocation in selection.pin_allocations}
    fixed = {pin.pin for pin in pins if pin.role == "mcu"}
    already_no_connect = {
        pin.pin for pin in no_connects if pin.role == "mcu"
    }
    no_connects.extend(
        NoConnect(role="mcu", pin=pin.pin)
        for pin in ESP32_S3_MINI_1_MINIMAL.allocatable_pins
        if pin.pin not in allocated | fixed | already_no_connect
    )
    definition = ESP32_S3_MINI_1_MINIMAL.model_copy(
        update={
            "parts": tuple(parts),
            "pins": tuple(pins),
            "no_connects": tuple(no_connects),
        }
    )
    return expand_static_definition(definition, resolved)
