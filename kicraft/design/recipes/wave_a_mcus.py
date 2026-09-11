"""Exact Wave-A MCU core recipes with reviewed package pin maps."""

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
)

_PASSIVE_CAP = ("Device:C", "Capacitor_SMD:C_0603_1608Metric")
_PASSIVE_RES = ("Device:R", "Resistor_SMD:R_0603_1608Metric")
_PROGRAM_HEADER = (
    "Connector_Generic:Conn_01x04",
    "Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical",
)
_REVIEW_DATE = "2026-09-09"
_OPTIONAL_USB_PINS: dict[str, tuple[str, str]] = {}
_AUTO_RESET_BOOT_HIGH_PINS: dict[str, tuple[str, ...]] = {}
_AUTO_RESET_SCHEMATIC = "https://dl.espressif.com/dl/schematics/esp32_devkitc_v4-sch-20180607a.pdf"
_AUTO_RESET_TRANSISTOR_DATASHEET = "https://www.onsemi.com/pdf/datasheet/mmbt3904lt1-d.pdf"


def _gpio_capabilities(*, adc: bool = True) -> tuple[str, ...]:
    values = [
        "gpio",
        "input",
        "output",
        "pwm",
        "interrupt",
        "i2c-sda",
        "i2c-scl",
        "spi-sclk",
        "spi-mosi",
        "spi-miso",
        "spi-cs",
        "uart-tx",
        "uart-rx",
    ]
    if adc:
        values.append("adc")
    return tuple(values)


def _module_recipe(
    *,
    recipe: str,
    family: str,
    exact_part: str,
    aliases: tuple[str, ...],
    symbol: str,
    footprint: str,
    datasheet: str,
    guideline: str,
    gpio_to_pin: dict[int, str],
    ground_pins: tuple[str, ...],
    vdd_pin: str,
    en_pin: str,
    boot_gpio: int,
    uart_tx_pin: str,
    uart_rx_pin: str,
    usb_pins: tuple[str, str] | None,
    strapping: frozenset[int],
    no_connect_pins: tuple[str, ...] = (),
    input_only_gpios: frozenset[int] = frozenset(),
    boot_high_gpios: frozenset[int] = frozenset(),
) -> RecipeDefinition:
    fixed = {boot_gpio}
    if usb_pins:
        fixed.update(gpio for gpio, pin in gpio_to_pin.items() if pin in usb_pins)
    internal_nets = ["en", "boot", "uart_tx", "uart_rx"]
    parts = [
        Group(
            role="mcu",
            reference_prefix="U",
            value=exact_part,
            symbol=symbol,
            footprint=footprint,
            sheet_role="mcu",
            mpn=exact_part,
            datasheet=datasheet,
        ),
        Group(
            role="bulk_decoupling",
            reference_prefix="C",
            value="10uF",
            symbol=_PASSIVE_CAP[0],
            footprint=_PASSIVE_CAP[1],
            sheet_role="mcu",
        ),
        Group(
            role="local_decoupling",
            reference_prefix="C",
            value="100nF",
            symbol=_PASSIVE_CAP[0],
            footprint=_PASSIVE_CAP[1],
            sheet_role="mcu",
        ),
        Group(
            role="en_capacitor",
            reference_prefix="C",
            value="1uF",
            symbol=_PASSIVE_CAP[0],
            footprint=_PASSIVE_CAP[1],
            sheet_role="mcu",
        ),
        Group(
            role="en_pullup",
            reference_prefix="R",
            value="10k",
            symbol=_PASSIVE_RES[0],
            footprint=_PASSIVE_RES[1],
            sheet_role="mcu",
        ),
        Group(
            role="boot_pullup",
            reference_prefix="R",
            value="10k",
            symbol=_PASSIVE_RES[0],
            footprint=_PASSIVE_RES[1],
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
            value="UART PROGRAM",
            symbol="Connector_Generic:Conn_01x06",
            footprint="Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical",
            sheet_role="mcu",
        ),
    ]
    pins = [
        Pin(role="mcu", pin=vdd_pin, net="vdd"),
        *(Pin(role="mcu", pin=pin, net="gnd") for pin in ground_pins),
        Pin(role="mcu", pin=en_pin, net="en"),
        Pin(role="mcu", pin=gpio_to_pin[boot_gpio], net="boot"),
        Pin(role="mcu", pin=uart_tx_pin, net="uart_tx"),
        Pin(role="mcu", pin=uart_rx_pin, net="uart_rx"),
        Pin(role="bulk_decoupling", pin="1", net="vdd"),
        Pin(role="bulk_decoupling", pin="2", net="gnd"),
        Pin(role="local_decoupling", pin="1", net="vdd"),
        Pin(role="local_decoupling", pin="2", net="gnd"),
        Pin(role="en_capacitor", pin="1", net="en"),
        Pin(role="en_capacitor", pin="2", net="gnd"),
        Pin(role="en_pullup", pin="1", net="vdd"),
        Pin(role="en_pullup", pin="2", net="en"),
        Pin(role="boot_pullup", pin="1", net="vdd"),
        Pin(role="boot_pullup", pin="2", net="boot"),
        Pin(role="reset_button", pin="1", net="en"),
        Pin(role="reset_button", pin="2", net="gnd"),
        Pin(role="boot_button", pin="1", net="boot"),
        Pin(role="boot_button", pin="2", net="gnd"),
        Pin(role="program_header", pin="1", net="vdd"),
        Pin(role="program_header", pin="2", net="gnd"),
        Pin(role="program_header", pin="3", net="uart_tx"),
        Pin(role="program_header", pin="4", net="uart_rx"),
        Pin(role="program_header", pin="5", net="en"),
        Pin(role="program_header", pin="6", net="boot"),
    ]
    ports = [
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="uart_tx", direction="output", required=False),
        Port(name="uart_rx", direction="input", required=False),
        Port(name="dtr_n", direction="input", required=False),
        Port(name="rts_n", direction="input", required=False),
    ]
    parameters: dict[str, bool] = {"auto_reset": False}
    allowed: dict[str, tuple[bool, ...]] = {"auto_reset": (False, True)}
    if usb_pins:
        ports.extend(
            (
                Port(name="usb_dm", direction="bidirectional", required=False),
                Port(name="usb_dp", direction="bidirectional", required=False),
            )
        )
        parameters["native_usb"] = False
        allowed["native_usb"] = (False, True)
        internal_nets.extend(("usb_dm", "usb_dp"))
        _OPTIONAL_USB_PINS[recipe] = usb_pins
    _AUTO_RESET_BOOT_HIGH_PINS[recipe] = tuple(
        gpio_to_pin[gpio] for gpio in sorted(boot_high_gpios)
    )
    return RecipeDefinition(
        recipe=recipe,
        family=family,
        exact_part=exact_part,
        maturity="production",
        protected_aliases=(family, exact_part),
        identity_aliases=aliases,
        required_sheet_roles=("mcu",),
        parameter_defaults=parameters,
        allowed_parameters=allowed,
        ports=tuple(ports),
        internal_nets=tuple(internal_nets),
        parts=tuple(parts),
        pins=tuple(pins),
        no_connects=tuple(NoConnect(role="mcu", pin=pin) for pin in no_connect_pins),
        allocatable_pins=tuple(
            AllocatablePin(
                role="mcu",
                pin=pin,
                gpio=gpio,
                capabilities=_gpio_capabilities(adc=gpio <= 20),
                strapping=gpio in strapping,
                reserved=gpio in fixed,
                input_only=gpio in input_only_gpios,
            )
            for gpio, pin in sorted(gpio_to_pin.items())
            if gpio not in fixed
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
        ),
        electrical_assertions=(
            Assertion(
                code="mcu_supply_3v3",
                message="Module supply remains within its 3.0 V to 3.6 V rating",
            ),
            Assertion(
                code="mcu_programming_access",
                message="UART, enable, boot, power, and ground reach a physical header",
            ),
        ),
        source_documents=(
            SourceDocument(
                url=datasheet,
                title=f"{exact_part} datasheet",
                revision="current",
                reviewed_date=_REVIEW_DATE,
                sections=("Pin Definitions", "Recommended PCB Land Pattern"),
            ),
            SourceDocument(
                url=guideline,
                title="Espressif Hardware Design Guidelines",
                revision="current",
                reviewed_date=_REVIEW_DATE,
                sections=("Power Supply", "Chip Power-up and Reset", "Strapping Pins"),
            ),
            SourceDocument(
                url=_AUTO_RESET_SCHEMATIC,
                title="Espressif ESP32-DevKitC V4 reference schematic",
                revision="2018-06-07",
                reviewed_date="2026-09-11",
                sections=("Auto program: cross-coupled NPN circuit and DTR/RTS truth table",),
            ),
            SourceDocument(
                url=_AUTO_RESET_TRANSISTOR_DATASHEET,
                title="onsemi MMBT3904L NPN transistor datasheet",
                revision="Rev 15",
                reviewed_date="2026-09-11",
                sections=("SOT-23: base 1, emitter 2, collector 3", "Electrical characteristics"),
            ),
            SourceDocument(
                url="https://docs.espressif.com/projects/esptool/en/latest/esp32c3/advanced-topics/boot-mode-selection.html",
                title="Espressif esptool automatic bootloader circuit",
                revision="current",
                reviewed_date="2026-09-11",
                sections=("Automatic Bootloader: active-low controls and 1uF-10uF EN capacitor",),
            ),
            *(
                (
                    SourceDocument(
                        url="https://dl.espressif.com/dl/schematics/SCH_ESP32-C3-DEVKITM-1_V1_20200915A.pdf",
                        title="Espressif ESP32-C3-DevKitM-1 reference schematic",
                        revision="2020-09-15",
                        reviewed_date="2026-09-11",
                        sections=("GPIO8 10k pullup; auto-program circuit and truth table",),
                    ),
                )
                if boot_high_gpios
                else ()
            ),
        ),
    )


ESP32_S3_WROOM_1_MINIMAL = _module_recipe(
    recipe="esp32-s3-wroom-1-minimal@1",
    family="esp32-s3-wroom-1-module",
    exact_part="ESP32-S3-WROOM-1-N8R8",
    aliases=("ESP32-S3-WROOM-1-N8R8", "esp32-s3-wroom-1:ESP32-S3-WROOM-1"),
    symbol="esp32-s3-wroom-1:ESP32-S3-WROOM-1",
    footprint="esp32-s3-wroom-1:WIRELM-SMD_ESP32-S3-WROOM-1",
    datasheet="https://www.espressif.com/sites/default/files/documentation/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf",
    guideline="https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32s3/schematic-checklist.html",
    gpio_to_pin={
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        15: "8",
        16: "9",
        17: "10",
        18: "11",
        8: "12",
        19: "13",
        20: "14",
        3: "15",
        46: "16",
        9: "17",
        10: "18",
        11: "19",
        12: "20",
        13: "21",
        14: "22",
        21: "23",
        47: "24",
        48: "25",
        45: "26",
        0: "27",
        35: "28",
        36: "29",
        37: "30",
        38: "31",
        39: "32",
        40: "33",
        41: "34",
        42: "35",
        2: "38",
        1: "39",
    },
    ground_pins=("1", "40", "41"),
    vdd_pin="2",
    en_pin="3",
    boot_gpio=0,
    uart_tx_pin="37",
    uart_rx_pin="36",
    usb_pins=("13", "14"),
    strapping=frozenset({0, 3, 45, 46}),
)

ESP32_WROOM_32E_MINIMAL = _module_recipe(
    recipe="esp32-wroom-32e-minimal@1",
    family="esp32-wroom-32e-module",
    exact_part="ESP32-WROOM-32E-N4",
    aliases=("ESP32-WROOM-32E-N4", "esp32-wroom-32e-n4:ESP32-WROOM-32E"),
    symbol="esp32-wroom-32e-n4:ESP32-WROOM-32E",
    footprint="esp32-wroom-32e-n4:WIFI-SMD_ESP32-WROOM-32E",
    datasheet="https://www.espressif.com/sites/default/files/documentation/esp32-wroom-32e_esp32-wroom-32ue_datasheet_en.pdf",
    guideline="https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32/schematic-checklist.html",
    gpio_to_pin={
        36: "4",
        39: "5",
        34: "6",
        35: "7",
        32: "8",
        33: "9",
        25: "10",
        26: "11",
        27: "12",
        14: "13",
        12: "14",
        13: "16",
        15: "23",
        2: "24",
        0: "25",
        4: "26",
        16: "27",
        17: "28",
        5: "29",
        18: "30",
        19: "31",
        21: "33",
        22: "36",
        23: "37",
    },
    ground_pins=("1", "15", "38", "39"),
    vdd_pin="2",
    en_pin="3",
    boot_gpio=0,
    uart_tx_pin="35",
    uart_rx_pin="34",
    usb_pins=None,
    strapping=frozenset({0, 2, 5, 12, 15}),
    input_only_gpios=frozenset({34, 35, 36, 39}),
    no_connect_pins=("17", "18", "19", "20", "21", "22", "32"),
)

ESP32_C3_MINI_1_MINIMAL = _module_recipe(
    recipe="esp32-c3-mini-1-minimal@1",
    family="esp32-c3-mini-1-module",
    exact_part="ESP32-C3-MINI-1-N4",
    aliases=("ESP32-C3-MINI-1-N4", "esp32-c3-mini-1-n4:ESP32-C3-MINI-1-N4"),
    symbol="esp32-c3-mini-1-n4:ESP32-C3-MINI-1-N4",
    footprint="esp32-c3-mini-1-n4:WIFIM-SMD_ESP32-C3-MINI-1",
    datasheet="https://www.espressif.com/sites/default/files/documentation/esp32-c3-mini-1_datasheet_en.pdf",
    guideline="https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32c3/schematic-checklist.html",
    gpio_to_pin={
        2: "5",
        3: "6",
        0: "12",
        1: "13",
        10: "16",
        4: "18",
        5: "19",
        6: "20",
        7: "21",
        8: "22",
        9: "23",
        18: "26",
        19: "27",
    },
    ground_pins=("1", "2", "11", "14", *(str(pin) for pin in range(36, 54))),
    vdd_pin="3",
    en_pin="8",
    boot_gpio=9,
    uart_tx_pin="31",
    uart_rx_pin="30",
    usb_pins=("26", "27"),
    strapping=frozenset({2, 8, 9}),
    boot_high_gpios=frozenset({8}),
    no_connect_pins=(
        "4",
        "7",
        "9",
        "10",
        "15",
        "17",
        "24",
        "25",
        "28",
        "29",
        "32",
        "33",
        "34",
        "35",
    ),
)


def _tinyavr_recipe(
    *,
    part: str,
    recipe: str,
    symbol: str,
    pins: dict[str, str],
    datasheet: str,
    touch_pins: tuple[str, ...] = (),
) -> RecipeDefinition:
    updi_pin = pins["PA0"]
    gpio_rows = [
        (name, pin) for name, pin in pins.items() if name.startswith(("PA", "PB")) and name != "PA0"
    ]
    return RecipeDefinition(
        recipe=recipe,
        family=part.lower().split("-")[0],
        exact_part=part,
        maturity="production",
        protected_aliases=("tinyAVR", part.split("-")[0]),
        identity_aliases=(part, symbol),
        required_sheet_roles=("mcu",),
        ports=(
            Port(name="vdd", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="updi", direction="bidirectional", required=False),
        ),
        internal_nets=("updi",),
        parts=(
            Group(
                role="mcu",
                reference_prefix="U",
                value=part,
                symbol=symbol,
                footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm"
                if len(pins) > 8
                else "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                sheet_role="mcu",
                mpn=part,
                datasheet=datasheet,
            ),
            Group(
                role="decoupling",
                reference_prefix="C",
                value="100nF",
                symbol=_PASSIVE_CAP[0],
                footprint=_PASSIVE_CAP[1],
                sheet_role="mcu",
            ),
            Group(
                role="updi_header",
                reference_prefix="J",
                value="UPDI",
                symbol=_PROGRAM_HEADER[0],
                footprint=_PROGRAM_HEADER[1],
                sheet_role="mcu",
            ),
        ),
        pins=(
            Pin(role="mcu", pin=pins["VDD"], net="vdd"),
            Pin(role="mcu", pin=pins["GND"], net="gnd"),
            Pin(role="mcu", pin=updi_pin, net="updi"),
            Pin(role="decoupling", pin="1", net="vdd"),
            Pin(role="decoupling", pin="2", net="gnd"),
            Pin(role="updi_header", pin="1", net="vdd"),
            Pin(role="updi_header", pin="2", net="updi"),
            Pin(role="updi_header", pin="3", net="gnd"),
        ),
        no_connects=(NoConnect(role="updi_header", pin="4"),),
        allocatable_pins=tuple(
            AllocatablePin(
                role="mcu",
                pin=pin,
                capabilities=_gpio_capabilities() + (("touch",) if name in touch_pins else ()),
            )
            for name, pin in gpio_rows
        ),
        placement_constraints=(
            PlacementConstraint(
                kind="decoupling_proximity",
                role="decoupling",
                parameters={"anchor_role": "mcu", "max_mm": 3.0},
            ),
        ),
        electrical_assertions=(
            Assertion(
                code="tinyavr_updi_access", message="PA0/UPDI reaches a physical programming header"
            ),
        ),
        source_documents=(
            SourceDocument(
                url=datasheet,
                title=f"{part} data sheet",
                revision="current",
                reviewed_date=_REVIEW_DATE,
                sections=("Pinout", "UPDI", "Hardware Guidelines"),
            ),
        ),
    )


ATTINY402_UPDI_MINIMAL = _tinyavr_recipe(
    part="ATTINY402-SSN",
    recipe="attiny402-updi-minimal@1",
    symbol="MCU_Microchip_ATtiny:ATtiny402-SS",
    pins={
        "VDD": "1",
        "PA6": "2",
        "PA7": "3",
        "PA1": "4",
        "PA2": "5",
        "PA3": "6",
        "PA0": "7",
        "GND": "8",
    },
    datasheet="https://ww1.microchip.com/downloads/aemDocuments/documents/MCU08/ProductDocuments/DataSheets/ATtiny202-204-402-404-406-DataSheet-DS40002318A.pdf",
)
ATTINY412_UPDI_MINIMAL = _tinyavr_recipe(
    part="ATTINY412-SSN",
    recipe="attiny412-updi-minimal@1",
    symbol="MCU_Microchip_ATtiny:ATtiny412-SS",
    pins={
        "VDD": "1",
        "PA6": "2",
        "PA7": "3",
        "PA1": "4",
        "PA2": "5",
        "PA3": "6",
        "PA0": "7",
        "GND": "8",
    },
    datasheet="https://ww1.microchip.com/downloads/aemDocuments/documents/MCU08/ProductDocuments/DataSheets/ATtiny212-214-412-414-416-DataSheet-DS40002287A.pdf",
)
ATTINY1614_UPDI_MINIMAL = _tinyavr_recipe(
    part="ATTINY1614-SSNR",
    recipe="attiny1614-updi-minimal@1",
    symbol="MCU_Microchip_ATtiny:ATtiny1614-SS",
    pins={
        "VDD": "1",
        "PA4": "2",
        "PA5": "3",
        "PA6": "4",
        "PA7": "5",
        "PB3": "6",
        "PB2": "7",
        "PB1": "8",
        "PB0": "9",
        "PA0": "10",
        "PA1": "11",
        "PA2": "12",
        "PA3": "13",
        "GND": "14",
    },
    datasheet="https://ww1.microchip.com/downloads/en/DeviceDoc/ATtiny1614-16-17-DataSheet-DS40002204A.pdf",
    # SOIC14 PTC Y-lines for self-capacitance; application firmware must configure
    # and link the QTouch Library (DS40002204A, sections 32.1 and 32.6).
    touch_pins=("PA4", "PA5", "PA6", "PA7", "PB1", "PB0"),
)
ATTINY1614_UPDI_MINIMAL = ATTINY1614_UPDI_MINIMAL.model_copy(
    update={
        "source_documents": (
            SourceDocument(
                url=ATTINY1614_UPDI_MINIMAL.parts[0].datasheet,
                title="ATtiny1614/1616/1617 data sheet",
                revision="DS40002204A",
                reviewed_date="2026-09-11",
                sections=(
                    "4.1 14-Pin SOIC",
                    "5.1 Table 5-1 PORT Function Multiplexing",
                    "32.1 PTC Overview",
                    "32.6 PTC Functional Description",
                    "33 UPDI",
                ),
            ),
        ),
    }
)


def _simple_programmed_mcu(
    *,
    recipe: str,
    family: str,
    part: str,
    symbol: str,
    footprint: str,
    datasheet: str,
    power_pins: tuple[str, ...],
    ground_pins: tuple[str, ...],
    program_pins: tuple[tuple[str, str], ...],
    gpio_pins: tuple[str, ...],
    extra_parts: tuple[Group, ...] = (),
    extra_pins: tuple[Pin, ...] = (),
    extra_ports: tuple[Port, ...] = (),
    internal_nets: tuple[str, ...] = (),
    pin_capabilities: dict[str, tuple[str, ...]] | None = None,
) -> RecipeDefinition:
    program_nets = tuple(dict.fromkeys(net for _pin, net in program_pins))
    return RecipeDefinition(
        recipe=recipe,
        family=family,
        exact_part=part,
        maturity="production",
        protected_aliases=(family, part),
        identity_aliases=(part, symbol),
        required_sheet_roles=("mcu",),
        ports=(
            Port(name="vdd", direction="power"),
            Port(name="gnd", direction="power"),
            *(Port(name=net, direction="bidirectional", required=False) for net in program_nets),
            *extra_ports,
        ),
        internal_nets=tuple(net for net in internal_nets if net not in program_nets),
        parts=(
            Group(
                role="mcu",
                reference_prefix="U",
                value=part,
                symbol=symbol,
                footprint=footprint,
                sheet_role="mcu",
                mpn=part,
                datasheet=datasheet,
            ),
            Group(
                role="decoupling",
                reference_prefix="C",
                quantity=max(1, len(power_pins)),
                value="100nF",
                symbol=_PASSIVE_CAP[0],
                footprint=_PASSIVE_CAP[1],
                sheet_role="mcu",
            ),
            Group(
                role="program_header",
                reference_prefix="J",
                value="PROGRAM",
                symbol=_PROGRAM_HEADER[0],
                footprint=_PROGRAM_HEADER[1],
                sheet_role="mcu",
            ),
            *extra_parts,
        ),
        pins=(
            *(Pin(role="mcu", pin=pin, net="vdd") for pin in power_pins),
            *(Pin(role="mcu", pin=pin, net="gnd") for pin in ground_pins),
            *(
                Pin(role="decoupling", index=index, pin="1", net="vdd")
                for index in range(max(1, len(power_pins)))
            ),
            *(
                Pin(role="decoupling", index=index, pin="2", net="gnd")
                for index in range(max(1, len(power_pins)))
            ),
            *(Pin(role="mcu", pin=pin, net=net) for pin, net in program_pins),
            *(
                Pin(role="program_header", index=0, pin=str(index + 1), net=net)
                for index, (_, net) in enumerate(program_pins)
            ),
            *extra_pins,
        ),
        no_connects=tuple(
            NoConnect(role="program_header", pin=str(index))
            for index in range(len(program_pins) + 1, 5)
        ),
        allocatable_pins=tuple(
            AllocatablePin(
                role="mcu",
                pin=pin,
                capabilities=(*_gpio_capabilities(), *(pin_capabilities or {}).get(pin, ())),
            )
            for pin in gpio_pins
            if pin not in {row[0] for row in program_pins}
            and pin not in {row.pin for row in extra_pins if row.role == "mcu"}
        ),
        placement_constraints=(
            PlacementConstraint(
                kind="decoupling_proximity",
                role="decoupling",
                parameters={"anchor_role": "mcu", "max_mm": 3.0},
            ),
        ),
        electrical_assertions=(
            Assertion(
                code="mcu_programming_access",
                message="The reviewed programming interface reaches a physical header",
            ),
        ),
        source_documents=(
            SourceDocument(
                url=datasheet,
                title=f"{part} data sheet",
                revision="current",
                reviewed_date=_REVIEW_DATE,
                sections=("Pinouts", "Power supply", "Programming"),
            ),
        ),
    )


STM32F103C8T6_MINIMAL = _simple_programmed_mcu(
    recipe="stm32f103c8t6-minimal@1",
    family="stm32f103c8",
    part="STM32F103C8T6",
    symbol="stm32f103c8t6:STM32F103C8T6",
    footprint="stm32f103c8t6:LQFP-48_L7.0-W7.0-P0.50-LS9.0-BL",
    datasheet="https://www.st.com/resource/en/datasheet/stm32f103c8.pdf",
    power_pins=("1", "9", "24", "36", "48"),
    ground_pins=("8", "23", "35", "47"),
    program_pins=(("34", "swdio"), ("37", "swclk"), ("7", "nrst"), ("44", "boot0")),
    gpio_pins=tuple(
        str(pin) for pin in range(2, 49) if pin not in {5, 6, 8, 9, 23, 24, 35, 36, 47, 48}
    ),
    # ST STM32F103x8/xB Table 5: LQFP48 PB8=45 CANRX, PB9=46 CANTX
    # (remapped functions). PA11/PA12 remain reserved by this recipe for USB.
    pin_capabilities={"45": ("can-rx",), "46": ("can-tx",)},
    internal_nets=("swdio", "swclk", "nrst", "boot0", "hse_in", "hse_out"),
    extra_ports=(
        Port(name="usb_dm", direction="bidirectional", required=False),
        Port(name="usb_dp", direction="bidirectional", required=False),
    ),
    extra_parts=(
        Group(
            role="boot0_pulldown",
            reference_prefix="R",
            value="10k",
            symbol=_PASSIVE_RES[0],
            footprint=_PASSIVE_RES[1],
            sheet_role="mcu",
        ),
        Group(
            role="reset_pullup",
            reference_prefix="R",
            value="10k",
            symbol=_PASSIVE_RES[0],
            footprint=_PASSIVE_RES[1],
            sheet_role="mcu",
        ),
        Group(
            role="hse_crystal",
            reference_prefix="Y",
            value="8MHz",
            symbol="Device:Crystal_GND24",
            footprint="Crystal:Crystal_SMD_3225-4Pin_3.2x2.5mm",
            sheet_role="mcu",
        ),
        Group(
            role="hse_caps",
            reference_prefix="C",
            quantity=2,
            value="18pF",
            symbol=_PASSIVE_CAP[0],
            footprint=_PASSIVE_CAP[1],
            sheet_role="mcu",
        ),
    ),
    extra_pins=(
        Pin(role="boot0_pulldown", pin="1", net="boot0"),
        Pin(role="boot0_pulldown", pin="2", net="gnd"),
        Pin(role="reset_pullup", pin="1", net="vdd"),
        Pin(role="reset_pullup", pin="2", net="nrst"),
        Pin(role="mcu", pin="5", net="hse_in"),
        Pin(role="mcu", pin="6", net="hse_out"),
        Pin(role="mcu", pin="32", net="usb_dm"),
        Pin(role="mcu", pin="33", net="usb_dp"),
        Pin(role="hse_crystal", pin="1", net="hse_in"),
        Pin(role="hse_crystal", pin="3", net="hse_out"),
        Pin(role="hse_crystal", pin="2", net="gnd"),
        Pin(role="hse_crystal", pin="4", net="gnd"),
        Pin(role="hse_caps", index=0, pin="1", net="hse_in"),
        Pin(role="hse_caps", index=0, pin="2", net="gnd"),
        Pin(role="hse_caps", index=1, pin="1", net="hse_out"),
        Pin(role="hse_caps", index=1, pin="2", net="gnd"),
    ),
)
STM32F103C8T6_MINIMAL = STM32F103C8T6_MINIMAL.model_copy(
    update={
        "parameter_defaults": {"can_remap": "pb8-pb9"},
        "allowed_parameters": {"can_remap": ("pb8-pb9",)},
        "source_documents": (
            *STM32F103C8T6_MINIMAL.source_documents,
            SourceDocument(
                url="https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/1139/STM32F103x8%2CB.pdf",
                title="ST STM32F103x8/xB datasheet (distributor mirror)",
                revision="Doc ID 13587 Rev 10",
                reviewed_date="2026-09-11",
                sections=("Table 5: PB8 pin 45 CANRX and PB9 pin 46 CANTX remap",),
            ),
            SourceDocument(
                url="https://abracon.com/Resonators/abm8.pdf",
                title="ABM8 four-pad ceramic crystal mechanical outline",
                revision="2020-07-29",
                reviewed_date="2026-09-11",
                sections=("Outline drawing: resonator terminals 1/3, grounded case 2/4",),
            ),
        ),
    }
)

CH32V003J4M6_MINIMAL = _simple_programmed_mcu(
    recipe="ch32v003j4m6-minimal@1",
    family="ch32v003",
    part="CH32V003J4M6",
    symbol="ch32v003j4m6:CH32V003J4M6",
    footprint="ch32v003j4m6:SOP-8_L4.9-W3.8-P1.27-LS6.0-BL",
    datasheet="https://www.wch-ic.com/downloads/CH32V003DS0_PDF.html",
    power_pins=("4",),
    ground_pins=("2",),
    program_pins=(("8", "swio"),),
    gpio_pins=("1", "3", "5", "6", "7"),
    internal_nets=("swio",),
)

WAVE_A_MCU_RECIPES = (
    ESP32_S3_WROOM_1_MINIMAL,
    ESP32_WROOM_32E_MINIMAL,
    ESP32_C3_MINI_1_MINIMAL,
    STM32F103C8T6_MINIMAL,
    ATTINY402_UPDI_MINIMAL,
    ATTINY412_UPDI_MINIMAL,
    ATTINY1614_UPDI_MINIMAL,
    CH32V003J4M6_MINIMAL,
)


def expand_wave_a_mcu(resolved):
    """Materialize only explicitly enabled USB and programming support circuits."""
    from .registry import expand_static_definition

    definition = next(
        definition
        for definition in WAVE_A_MCU_RECIPES
        if definition.recipe == resolved.selection.recipe
    )
    pins = list(definition.pins)
    parts = list(definition.parts)
    internal_nets = list(definition.internal_nets)
    no_connects = list(definition.no_connects)
    bindings = resolved.selection.port_bindings
    if "auto_reset" in definition.parameter_defaults:
        auto_reset = resolved.parameters["auto_reset"]
        if auto_reset and not {"uart_tx", "uart_rx", "dtr_n", "rts_n"} <= bindings.keys():
            raise ValueError(
                f"recipe {definition.recipe} auto_reset=True requires uart_tx, uart_rx, dtr_n and rts_n bindings"
            )
        if not auto_reset and {"dtr_n", "rts_n"} & bindings.keys():
            raise ValueError(
                f"recipe {definition.recipe} dtr_n/rts_n bindings require auto_reset=True"
            )
        if auto_reset:
            if len({bindings[name] for name in ("uart_tx", "uart_rx", "dtr_n", "rts_n")}) != 4:
                raise ValueError(f"recipe {definition.recipe} programming signals must be distinct")
            # Reference truth table (physical levels): DTR RTS -> EN BOOT:
            # 00 -> 11, 11 -> 11, 10 -> 01, 01 -> 10.
            # Each emitter uses the opposite handshake, NOT ground. 10k
            # base resistors and the existing EN 10k/1uF RC retain timing.
            # MMBT3904LT1G: onsemi SOT-23 B=1, E=2, C=3; KiCad's
            # Transistor_BJT:MMBT3904 inherits that exact Q_NPN_BEC map.
            for target, base_control, emitter_control in (
                ("en", "dtr_n", "rts_n"),
                ("boot", "rts_n", "dtr_n"),
            ):
                transistor = f"auto_reset_{target}"
                resistor = f"{transistor}_base_resistor"
                base = f"{transistor}_base"
                internal_nets.append(base)
                parts.extend(
                    (
                        Group(
                            role=transistor,
                            reference_prefix="Q",
                            value="MMBT3904",
                            symbol="Transistor_BJT:MMBT3904",
                            footprint="Package_TO_SOT_SMD:SOT-23",
                            sheet_role="mcu",
                            mpn="MMBT3904LT1G",
                            datasheet=_AUTO_RESET_TRANSISTOR_DATASHEET,
                        ),
                        Group(
                            role=resistor,
                            reference_prefix="R",
                            value="10k",
                            symbol=_PASSIVE_RES[0],
                            footprint=_PASSIVE_RES[1],
                            sheet_role="mcu",
                        ),
                    )
                )
                pins.extend(
                    (
                        Pin(role=resistor, pin="1", net=base_control),
                        Pin(role=resistor, pin="2", net=base),
                        Pin(role=transistor, pin="1", net=base),
                        Pin(role=transistor, pin="2", net=emitter_control),
                        Pin(role=transistor, pin="3", net=target),
                    )
                )
            # ESP32-C3 additionally samples GPIO8 high for serial download.
            # Its pullup shares any allocated application conductor; it does
            # not reserve/reassign that GPIO or add a second MCU pin owner.
            for pin in _AUTO_RESET_BOOT_HIGH_PINS[definition.recipe]:
                allocation = next(
                    (row for row in resolved.selection.pin_allocations if row.pin == pin),
                    None,
                )
                net = allocation.net if allocation else f"auto_reset_boot_high_{pin}"
                role = f"auto_reset_boot_high_pullup_{pin}"
                parts.append(
                    Group(
                        role=role,
                        reference_prefix="R",
                        value="10k",
                        symbol=_PASSIVE_RES[0],
                        footprint=_PASSIVE_RES[1],
                        sheet_role="mcu",
                    )
                )
                pins.extend(
                    (
                        Pin(role=role, pin="1", net="vdd"),
                        Pin(role=role, pin="2", net=net),
                    )
                )
                if allocation is None:
                    internal_nets.append(net)
                    pins.append(Pin(role="mcu", pin=pin, net=net))
    usb_pins = _OPTIONAL_USB_PINS.get(definition.recipe)
    if usb_pins is not None:
        if resolved.parameters["native_usb"]:
            if not {"usb_dm", "usb_dp"} <= bindings.keys():
                raise ValueError(
                    f"recipe {definition.recipe} native_usb=True requires usb_dm and usb_dp bindings"
                )
            pins.extend(
                (
                    Pin(role="mcu", pin=usb_pins[0], net="usb_dm"),
                    Pin(role="mcu", pin=usb_pins[1], net="usb_dp"),
                )
            )
        else:
            no_connects.extend(
                (
                    NoConnect(role="mcu", pin=usb_pins[0]),
                    NoConnect(role="mcu", pin=usb_pins[1]),
                )
            )
    return expand_static_definition(
        definition.model_copy(
            update={
                "parts": tuple(parts),
                "pins": tuple(pins),
                "internal_nets": tuple(internal_nets),
                "no_connects": tuple(no_connects),
            }
        ),
        resolved,
    )
