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
# Module recipes with a factory-supported native USB pair (S3-WROOM, C3).
_NATIVE_USB_PINS: dict[str, tuple[str, str]] = {}
# Fixed UART0 pins for those native modules, marked no-connect when unbound.
_NATIVE_USB_UART_PINS: dict[str, tuple[str, str]] = {}
_DOWNLOAD_BOOT_HIGH_PINS: dict[str, tuple[str, ...]] = {}
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
    adc_gpios: frozenset[int],
    adc_input_max_v: float,
    ground_pins: tuple[str, ...],
    vdd_pin: str,
    en_pin: str,
    boot_gpio: int,
    uart_tx_pin: str,
    uart_rx_pin: str,
    usb_pins: tuple[str, str] | None,
    strapping: frozenset[int],
    adc_datasheet: str | None = None,
    no_connect_pins: tuple[str, ...] = (),
    input_only_gpios: frozenset[int] = frozenset(),
    boot_high_gpios: frozenset[int] = frozenset(),
    family_aliases: tuple[str, ...] = (),
 ) -> RecipeDefinition:
    fixed = {boot_gpio}
    native_usb = usb_pins is not None
    if usb_pins:
        fixed.update(gpio for gpio, pin in gpio_to_pin.items() if pin in usb_pins)
    internal_nets = ["en", "boot"] if native_usb else ["en", "boot", "uart_tx", "uart_rx"]
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
            symbol="kmr232ngulclfs:KMR232NGULCLFS",
            footprint="kmr232ngulclfs:KMR232NGULCLFS",
            sheet_role="mcu",
            mpn="KMR232NGULCLFS",
        ),
        Group(
            role="boot_button",
            reference_prefix="SW",
            value="BOOT",
            symbol="kmr232ngulclfs:KMR232NGULCLFS",
            footprint="kmr232ngulclfs:KMR232NGULCLFS",
            sheet_role="mcu",
            mpn="KMR232NGULCLFS",
        ),
        *(
            ()
            if native_usb
            else (
                Group(
                    role="program_header",
                    reference_prefix="J",
                    value="UART PROGRAM",
                    symbol="Connector_Generic:Conn_01x06",
                    footprint="Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical",
                    sheet_role="mcu",
                ),
            )
        ),
    ]
    pins = [
        Pin(role="mcu", pin=vdd_pin, net="vdd"),
        *(Pin(role="mcu", pin=pin, net="gnd") for pin in ground_pins),
        Pin(role="mcu", pin=en_pin, net="en"),
        Pin(role="mcu", pin=gpio_to_pin[boot_gpio], net="boot"),
        *(
            ()
            if native_usb
            else (
                Pin(role="mcu", pin=uart_tx_pin, net="uart_tx"),
                Pin(role="mcu", pin=uart_rx_pin, net="uart_rx"),
            )
        ),
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
        *(
            ()
            if native_usb
            else (
                Pin(role="program_header", pin="1", net="vdd"),
                Pin(role="program_header", pin="2", net="gnd"),
                Pin(role="program_header", pin="3", net="uart_tx"),
                Pin(role="program_header", pin="4", net="uart_rx"),
                Pin(role="program_header", pin="5", net="en"),
                Pin(role="program_header", pin="6", net="boot"),
            )
        ),
    ]
    ports = [
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="uart_tx", direction="output", required=False),
        Port(name="uart_rx", direction="input", required=False),
        Port(name="dtr_n", direction="input", required=False),
        Port(name="rts_n", direction="input", required=False),
    ]
    parameters: dict[str, bool | float] = {
        "auto_reset": False, "consumer_input_max_v": adc_input_max_v,
    }
    allowed: dict[str, tuple[bool | float, ...]] = {
        "auto_reset": (False, True), "consumer_input_max_v": (adc_input_max_v,),
    }
    if usb_pins:
        ports.extend(
            (
                Port(name="usb_dm", direction="bidirectional", required=False),
                Port(name="usb_dp", direction="bidirectional", required=False),
            )
        )
        parameters["native_usb"] = False
        allowed["native_usb"] = (False, True)
        _NATIVE_USB_PINS[recipe] = usb_pins
        _NATIVE_USB_UART_PINS[recipe] = (uart_tx_pin, uart_rx_pin)
    _DOWNLOAD_BOOT_HIGH_PINS[recipe] = tuple(
        gpio_to_pin[gpio] for gpio in sorted(boot_high_gpios)
    )
    return RecipeDefinition(
        recipe=recipe,
        family=family,
        exact_part=exact_part,
        maturity="production",
        protected_aliases=(family, exact_part, *family_aliases),
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
                capabilities=_gpio_capabilities(adc=gpio in adc_gpios),
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
                message=(
                    "Native USB D-/D+ reach one physical USB data connector, "
                    "with enable and boot recovery access"
                    if native_usb
                    else "UART, enable, boot, power, and ground reach a physical header"
                ),
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
                url=adc_datasheet or datasheet,
                title="Espressif ADC calibrated measurement range",
                revision="current",
                reviewed_date=_REVIEW_DATE,
                sections=("ADC Calibration Results: ATTEN3 effective measurement range",),
            ),
            SourceDocument(
                url=guideline,
                title="Espressif Hardware Design Guidelines",
                revision="current",
                reviewed_date=_REVIEW_DATE,
                sections=("Power Supply", "Chip Power-up and Reset", "Strapping Pins", "USB"),
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


# Datasheet v1.7 Tables 1-1 and 3-1: the U N4R2 has 4 MB flash and
# 2 MB PSRAM; IO26 is used internally. GPIO45 is left unconnected so its
# internal pull-down selects the 3.3 V VDD_SPI default (section 4.2).
_S3U_RECIPE = "esp32-s3-mini-1u-n4r2-minimal@1"
ESP32_S3_MINI_1U_N4R2_MINIMAL = _module_recipe(
    recipe=_S3U_RECIPE,
    family="esp32-s3-module",
    exact_part="ESP32-S3-MINI-1U-N4R2",
    aliases=("ESP32-S3-MINI-1U-N4R2", "esp32-s3-mini-1u-n4r2:ESP32-S3-MINI-1U-N4R2"),
    symbol="esp32-s3-mini-1u-n4r2:ESP32-S3-MINI-1U-N4R2",
    footprint="esp32-s3-mini-1u-n4r2:BULETM-SMD_ESPRESSIF_ESP32-S3-MINI-1U-N8",
    datasheet="https://documentation.espressif.com/esp32-s3-mini-1_mini-1u_datasheet_en.pdf",
    guideline="https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32s3/schematic-checklist.html",
    gpio_to_pin={
        **{gpio: str(gpio + 4) for gpio in range(22)},
        47: "27", 33: "28", 34: "29", 48: "30", 35: "31",
        36: "32", 37: "33", 38: "34", 39: "35", 40: "36",
        41: "37", 42: "38",
    },
    adc_gpios=frozenset(range(1, 21)),
    adc_input_max_v=2.9,
    adc_datasheet="https://documentation.espressif.com/esp32-s3_datasheet_en.pdf",
    ground_pins=("1", "2", "42", "43", *(str(pin) for pin in range(46, 66))),
    vdd_pin="3",
    en_pin="45",
    boot_gpio=0,
    uart_tx_pin="39",
    uart_rx_pin="40",
    usb_pins=("23", "24"),
    strapping=frozenset({0, 3, 45, 46}),
    no_connect_pins=("7", "26", "41", "44"),
)
ESP32_S3_MINI_1U_N4R2_MINIMAL = ESP32_S3_MINI_1U_N4R2_MINIMAL.model_copy(update={
    "default_for_family": True,
    "protected_aliases": ("ESP32-S3", "ESP32-S3 module", "ESP32-S3-MINI-1U-N4R2"),
    "parameter_defaults": {**ESP32_S3_MINI_1U_N4R2_MINIMAL.parameter_defaults, "native_usb": True},
    "allowed_parameters": {**ESP32_S3_MINI_1U_N4R2_MINIMAL.allowed_parameters, "native_usb": (True,)},
    "ports": tuple(
        port.model_copy(update={"required": True}) if port.name in {"usb_dm", "usb_dp"} else port
        for port in ESP32_S3_MINI_1U_N4R2_MINIMAL.ports
    ),
    "allocatable_pins": tuple(
        pin for pin in ESP32_S3_MINI_1U_N4R2_MINIMAL.allocatable_pins if pin.gpio != 3
    ),
    # The U variant has an external RF connector, not a PCB antenna.
    "placement_constraints": tuple(
        rule for rule in ESP32_S3_MINI_1U_N4R2_MINIMAL.placement_constraints
        if rule.kind != "antenna_keepout"
    ) + tuple(
        PlacementConstraint(kind="owned_bypass", role=role, parameters={
            "anchor_role": "mcu", "power_net": "vdd", "ground_net": "gnd", "max_mm": 3.0,
        })
        for role in ("local_decoupling", "bulk_decoupling")
    ),
})


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
        38: "31",
        39: "32",
        40: "33",
        41: "34",
        42: "35",
        2: "38",
        1: "39",
    },
    adc_gpios=frozenset(range(1, 21)),
    adc_input_max_v=2.9,
    adc_datasheet="https://documentation.espressif.com/esp32-s3_datasheet_en.pdf",
    ground_pins=("1", "40", "41"),
    vdd_pin="2",
    en_pin="3",
    boot_gpio=0,
    uart_tx_pin="37",
    uart_rx_pin="36",
    usb_pins=("13", "14"),
    strapping=frozenset({0, 3, 45, 46}),
    # N8R8 uses these module pads for its internal Octal SPI PSRAM.
    no_connect_pins=("28", "29", "30"),
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
    adc_gpios=frozenset({0, 2, 4, 12, 13, 14, 15, 25, 26, 27, 32, 33, 34, 35, 36, 39}),
    adc_input_max_v=2.45,
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
    family_aliases=("ESP32-C3 module",),
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
    # GPIO5 is ADC2, whose unstable conversion results are excluded by ESP-IDF.
    adc_gpios=frozenset(range(5)),
    adc_input_max_v=2.5,
    adc_datasheet="https://documentation.espressif.com/esp32-c3_datasheet_en.pdf",
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

# The two supply capacitors belong to this module, not any other 3V3 load.
# Espressif's schematic checklist calls for local supply decoupling.
ESP32_C3_MINI_1_MINIMAL = ESP32_C3_MINI_1_MINIMAL.model_copy(update={
    "placement_constraints": ESP32_C3_MINI_1_MINIMAL.placement_constraints + tuple(
        PlacementConstraint(kind="owned_bypass", role=role, parameters={
            "anchor_role": "mcu", "power_net": "vdd", "ground_net": "gnd", "max_mm": 3.0,
        })
        for role in ("local_decoupling", "bulk_decoupling")
    ),
})

# The H4X is a separate orderable identity, not a replacement for an exact N4
# request. Espressif v2.2 table 1-1 specifies the same 4 MB flash and PCB
# antenna, with chip revision v1.1. LCSC C41349510 supplies its own CAD; its
# numbered pins and 0.8 mm-pitch 0.4 x 0.8 mm perimeter lands match the
# Espressif pin table and figure 11-1. RF performance and assembly remain
# product-level obligations.
_C3_H4X_RECIPE = "esp32-c3-mini-1-h4x-minimal@1"
ESP32_C3_MINI_1_H4X_MINIMAL = ESP32_C3_MINI_1_MINIMAL.model_copy(
    update={
        "recipe": _C3_H4X_RECIPE,
        "exact_part": "ESP32-C3-MINI-1-H4X",
        "default_for_family": True,
        "protected_aliases": ("esp32-c3-mini-1-module", "ESP32-C3", "ESP32-C3 module", "ESP32-C3-MINI-1-H4X"),
        "identity_aliases": ("ESP32-C3-MINI-1-H4X", "esp32-c3-mini-1-h4x:ESP32-C3-MINI-1-H4X"),
        "parts": (
            ESP32_C3_MINI_1_MINIMAL.parts[0].model_copy(
                update={
                    "value": "ESP32-C3-MINI-1-H4X",
                    "mpn": "ESP32-C3-MINI-1-H4X",
                    "symbol": "esp32-c3-mini-1-h4x:ESP32-C3-MINI-1-H4X",
                    "footprint": "esp32-c3-mini-1-h4x:WIFIM-SMD_ESP32-C3-MINI-1",
                    "datasheet": "https://documentation.espressif.com/esp32-c3_mini_1_datasheet_en.pdf",
                }
            ),
            *ESP32_C3_MINI_1_MINIMAL.parts[1:],
        ),
        "parameter_defaults": {
            **ESP32_C3_MINI_1_MINIMAL.parameter_defaults,
            "native_usb": True,
        },
        "allowed_parameters": {
            **ESP32_C3_MINI_1_MINIMAL.allowed_parameters,
            "native_usb": (True,),
        },
        "ports": tuple(
            port.model_copy(update={"required": True})
            if port.name in {"usb_dm", "usb_dp"} else port
            for port in ESP32_C3_MINI_1_MINIMAL.ports
        ),
        "source_documents": (
            SourceDocument(
                url="https://documentation.espressif.com/esp32-c3_mini_1_datasheet_en.pdf",
                title="ESP32-C3-MINI-1 and MINI-1U Datasheet",
                revision="v2.2",
                reviewed_date="2026-11-03",
                sections=("Table 1-1", "Table 3-1", "Figure 11-1"),
            ),
            SourceDocument(
                url="https://www.lcsc.com/product-detail/C41349510.html",
                title="LCSC ESP32-C3-MINI-1-H4X supplier identity and CAD",
                revision="C41349510",
                reviewed_date="2026-11-03",
                sections=("Manufacturer part identity", "Supplier symbol and land pattern"),
            ),
            *ESP32_C3_MINI_1_MINIMAL.source_documents[1:],
        ),
    }
)
_NATIVE_USB_PINS[_C3_H4X_RECIPE] = _NATIVE_USB_PINS[ESP32_C3_MINI_1_MINIMAL.recipe]
_NATIVE_USB_UART_PINS[_C3_H4X_RECIPE] = _NATIVE_USB_UART_PINS[ESP32_C3_MINI_1_MINIMAL.recipe]
_DOWNLOAD_BOOT_HIGH_PINS[_C3_H4X_RECIPE] = _DOWNLOAD_BOOT_HIGH_PINS[ESP32_C3_MINI_1_MINIMAL.recipe]

def _rom_uart_variant(base: RecipeDefinition, recipe: str, tx: str, rx: str,
                      usb_pins: tuple[str, str]) -> RecipeDefinition:
    """Canary manual-download construction for the same exact module.

    The separate recipe has no USB ports: it cannot realize a native USB
    contract.  Keep the USB recipe's reviewed module, support and placement
    constraints, and expose the ROM UART and both recovery controls.
    """
    _DOWNLOAD_BOOT_HIGH_PINS[recipe] = _DOWNLOAD_BOOT_HIGH_PINS[base.recipe]
    return base.model_copy(update={
        "recipe": recipe,
        # Registry exact selectors remain owned by the USB recipe. The
        # physical module's exact MPN and geometry are retained in parts.
        "family": base.family,
        "exact_part": None,
        "default_for_family": False,
        "maturity": "canary",
        "protected_aliases": (),
        "identity_aliases": (),
        "parameter_defaults": {k: v for k, v in base.parameter_defaults.items() if k != "native_usb"},
        "allowed_parameters": {k: v for k, v in base.allowed_parameters.items() if k != "native_usb"},
        "ports": tuple(port for port in base.ports if port.name not in {"usb_dm", "usb_dp"}),
        "internal_nets": (*base.internal_nets, "uart_tx", "uart_rx"),
        "parts": (*base.parts, Group(
            role="program_header", reference_prefix="J", value="UART PROGRAM",
            symbol="Connector_Generic:Conn_01x06",
            footprint="amphenol-68000-406hlf:HDR-TH_6P-P2.54-V-M-1",
            sheet_role="mcu", mpn="68000-406HLF",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588895884876242944",
        )),
        "pins": (*base.pins,
            Pin(role="mcu", pin=tx, net="uart_tx"),
            Pin(role="mcu", pin=rx, net="uart_rx"),
            *(Pin(role="program_header", pin=str(index), net=net) for index, net in enumerate(
                ("vdd", "gnd", "uart_tx", "uart_rx", "en", "boot"), start=1
            )),
        ),
        "no_connects": (*base.no_connects, *(NoConnect(role="mcu", pin=pin) for pin in usb_pins)),
        "electrical_assertions": tuple(
            assertion.model_copy(update={"message": "ROM UART, enable, boot, power, and ground reach a physical header"})
            if assertion.code == "mcu_programming_access" else assertion
            for assertion in base.electrical_assertions
        ),
        "source_documents": (*base.source_documents, SourceDocument(
            url="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588895884876242944",
            title="Amphenol FCI 68000 BERGSTIK II single-row header",
            revision="DC", reviewed_date="2026-11-03",
            sections=("Sheets 1-2: six positions, 2.54 mm pitch, 1.02 +/- 0.08 mm recommended holes",),
        )),
    })


ESP32_S3_MINI_1U_N4R2_UART = _rom_uart_variant(
    ESP32_S3_MINI_1U_N4R2_MINIMAL, "esp32-s3-mini-1u-n4r2-uart-minimal@1",
    "39", "40", ("23", "24"),
).model_copy(update={"maturity": "production"})
ESP32_C3_MINI_1_H4X_UART = _rom_uart_variant(
    ESP32_C3_MINI_1_H4X_MINIMAL, "esp32-c3-mini-1-h4x-uart-minimal@1",
    "31", "30", ("26", "27"),
)

# These same-value supports jointly fit the RF-edge owner's unchanged 3 mm
# supply/return bounds; two 0603 courtyards do not. Independent native, cold-route,
# full-DRC and fabrication review qualifies this bounded UART construction only.
_C3_UART_LOCAL_SUPPORT = {
    "bulk_decoupling": (
        "GRM155R61A106ME11D",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588893771849748480",
    ),
    "local_decoupling": (
        "CL05A104KA5NNNC",
        "https://product.samsungsem.com/mlcc/CL05A104KA5NNN.do",
    ),
}
ESP32_C3_MINI_1_H4X_UART = ESP32_C3_MINI_1_H4X_UART.model_copy(update={
    "maturity": "production",
    "parts": tuple(
        part.model_copy(update={
            "mpn": _C3_UART_LOCAL_SUPPORT[part.role][0],
            "footprint": "Capacitor_SMD:C_0402_1005Metric",
            "datasheet": _C3_UART_LOCAL_SUPPORT[part.role][1],
        }) if part.role in _C3_UART_LOCAL_SUPPORT else part
        for part in ESP32_C3_MINI_1_H4X_UART.parts
    ),
})



def _tinyavr_recipe(
    *,
    part: str,
    recipe: str,
    symbol: str,
    pins: dict[str, str],
    datasheet: str,
    touch_pins: tuple[str, ...] = (),
    footprint: str | None = None,
    default_for_family: bool = False,
) -> RecipeDefinition:
    updi_pin = pins["PA0"]
    gpio_rows = [
        (name, pin) for name, pin in pins.items() if name.startswith(("PA", "PB")) and name != "PA0"
    ]
    return RecipeDefinition(
        recipe=recipe,
        family=part.lower().split("-")[0],
        exact_part=part,
        default_for_family=default_for_family,
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
                footprint=footprint or (
                    "Package_SO:SOIC-14_3.9x8.7mm_P1.27mm"
                    if len(pins) > 8
                    else "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"
                ),
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


_ATTINY402_PINS = {
    "VDD": "1",
    "PA6": "2",
    "PA7": "3",
    "PA1": "4",
    "PA2": "5",
    "PA3": "6",
    "PA0": "7",
    "GND": "8",
}

ATTINY402_UPDI_MINIMAL = _tinyavr_recipe(
    part="ATTINY402-SSN",
    recipe="attiny402-updi-minimal@1",
    symbol="MCU_Microchip_ATtiny:ATtiny402-SS",
    pins=_ATTINY402_PINS,
    datasheet="https://ww1.microchip.com/downloads/aemDocuments/documents/MCU08/ProductDocuments/DataSheets/ATtiny202-204-402-404-406-DataSheet-DS40002318A.pdf",
)
ATTINY402_SSNR_UPDI_MINIMAL = _tinyavr_recipe(
    part="ATTINY402-SSNR",
    recipe="attiny402-ssnr-updi-minimal@1",
    symbol="attiny402-ssnr:ATTINY402-SSNR",
    footprint="attiny402-ssnr:SOIC-8_L5.0-W4.0-P1.27-LS6.0-BL",
    pins=_ATTINY402_PINS,
    datasheet="https://ww1.microchip.com/downloads/aemDocuments/documents/MCU08/ProductDocuments/DataSheets/ATtiny202-204-402-404-406-DataSheet-DS40002318A.pdf",
    default_for_family=True,
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


_STM32G030F6P6_DATASHEET = "https://www.st.com/resource/en/datasheet/stm32g030f6.pdf"
_ATTINY1604_DATASHEET = (
    "https://ww1.microchip.com/downloads/en/DeviceDoc/"
    "ATtiny804-06-07-1604-06-07-DataSheet-DS40002312A.pdf"
)


STM32G030F6P6_MINIMAL = RecipeDefinition(
    recipe="stm32g030f6p6-minimal@1",
    family="stm32g0",
    exact_part="STM32G030F6P6",
    default_for_family=True,
    maturity="production",
    protected_aliases=("STM32G0", "STM32G030"),
    identity_aliases=("stm32g030f6p6", "MCU_ST_STM32G0:STM32G030F6Px"),
    required_sheet_roles=("mcu",),
    parameter_defaults={
        "logic_voltage_v": 3.3,
        "clock_source": "hsi16_pll",
        "adc_expansion": "none",
        "gpio_expansion": "none",
        "consumer_input_max_v": 3.3,
    },
    allowed_parameters={
        "logic_voltage_v": (3.3,),
        "clock_source": ("hsi16_pll",),
        "adc_expansion": ("none", "ads1115x2"),
        "gpio_expansion": ("none", "mcp23017"),
        "consumer_input_max_v": (3.3,),
    },
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="swdio", direction="bidirectional", required=False),
        Port(name="swclk_boot0", direction="bidirectional", required=False),
        Port(name="nrst", direction="bidirectional", required=False),
        Port(name="uart_tx", direction="output", required=False),
        Port(name="uart_rx", direction="input", required=False),
        Port(name="i2c_scl", direction="bidirectional", required=False),
        Port(name="i2c_sda", direction="bidirectional", required=False),
        Port(name="spi_sck", direction="output", required=False),
        Port(name="spi_mosi", direction="output", required=False),
        Port(name="spi_miso", direction="input", required=False),
        Port(name="spi_cs0", direction="output", required=False),
        *(Port(name=f"adc_{index}", direction="input", required=False) for index in range(8)),
        *(Port(name=f"output_{index}", direction="output", required=False) for index in range(16)),
        *(Port(name=f"parallel_{index}", direction="output", required=False) for index in range(16)),
    ),
    fixed_port_pins={
        "uart_tx": "9",
        "uart_rx": "10",
        "i2c_scl": "1",
        "i2c_sda": "2",
        "spi_sck": "12",
        "spi_mosi": "14",
        "spi_miso": "13",
        "spi_cs0": "11",
    },
    internal_nets=("nrst", "swclk_boot0"),
    parts=(
        Group(
            role="mcu",
            reference_prefix="U",
            value="STM32G030F6P6",
            symbol="MCU_ST_STM32G0:STM32G030F6Px",
            footprint="Package_SO:TSSOP-20_4.4x6.5mm_P0.65mm",
            sheet_role="mcu",
            mpn="STM32G030F6P6",
            datasheet=_STM32G030F6P6_DATASHEET,
        ),
        Group(role="local_decoupling", reference_prefix="C", value="100nF", symbol=_PASSIVE_CAP[0], footprint=_PASSIVE_CAP[1], sheet_role="mcu"),
        Group(role="bulk_decoupling", reference_prefix="C", value="4.7uF", symbol=_PASSIVE_CAP[0], footprint=_PASSIVE_CAP[1], sheet_role="mcu"),
        Group(role="reset_capacitor", reference_prefix="C", value="100nF", symbol=_PASSIVE_CAP[0], footprint=_PASSIVE_CAP[1], sheet_role="mcu"),
        Group(role="reset_pullup", reference_prefix="R", value="10k", symbol=_PASSIVE_RES[0], footprint=_PASSIVE_RES[1], sheet_role="mcu"),
        Group(role="boot0_pulldown", reference_prefix="R", value="10k", symbol=_PASSIVE_RES[0], footprint=_PASSIVE_RES[1], sheet_role="mcu"),
        Group(role="reset_button", reference_prefix="SW", value="RESET", symbol="kmr232ngulclfs:KMR232NGULCLFS", footprint="kmr232ngulclfs:KMR232NGULCLFS", sheet_role="mcu", mpn="KMR232NGULCLFS"),
        Group(role="swd_header", reference_prefix="J", value="SWD", symbol="Connector_Generic:Conn_01x06", footprint="Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical", sheet_role="mcu"),
    ),
    pins=(
        Pin(role="mcu", pin="4", net="vdd"),
        Pin(role="mcu", pin="5", net="gnd"),
        Pin(role="mcu", pin="6", net="nrst"),
        Pin(role="mcu", pin="18", net="swdio"),
        Pin(role="mcu", pin="19", net="swclk_boot0"),
        Pin(role="local_decoupling", pin="1", net="vdd"),
        Pin(role="local_decoupling", pin="2", net="gnd"),
        Pin(role="bulk_decoupling", pin="1", net="vdd"),
        Pin(role="bulk_decoupling", pin="2", net="gnd"),
        Pin(role="reset_capacitor", pin="1", net="nrst"),
        Pin(role="reset_capacitor", pin="2", net="gnd"),
        Pin(role="reset_pullup", pin="1", net="vdd"),
        Pin(role="reset_pullup", pin="2", net="nrst"),
        Pin(role="boot0_pulldown", pin="1", net="swclk_boot0"),
        Pin(role="boot0_pulldown", pin="2", net="gnd"),
        Pin(role="reset_button", pin="1", net="nrst"),
        Pin(role="reset_button", pin="2", net="gnd"),
        Pin(role="swd_header", pin="1", net="vdd"),
        Pin(role="swd_header", pin="2", net="gnd"),
        Pin(role="swd_header", pin="3", net="swdio"),
        Pin(role="swd_header", pin="4", net="swclk_boot0"),
        Pin(role="swd_header", pin="5", net="nrst"),
    ),
    no_connects=(NoConnect(role="swd_header", pin="6"),),
    allocatable_pins=(
        AllocatablePin(role="mcu", pin="1", capabilities=("gpio", "input", "output", "interrupt", "i2c-scl")),
        AllocatablePin(role="mcu", pin="2", capabilities=("gpio", "input", "output", "interrupt", "i2c-sda")),
        AllocatablePin(role="mcu", pin="7", gpio=0, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc")),
        AllocatablePin(role="mcu", pin="8", gpio=1, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc")),
        AllocatablePin(role="mcu", pin="9", gpio=2, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "uart-tx")),
        AllocatablePin(role="mcu", pin="10", gpio=3, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "uart-rx")),
        AllocatablePin(role="mcu", pin="11", gpio=4, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "spi-cs")),
        AllocatablePin(role="mcu", pin="12", gpio=5, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "spi-sclk")),
        AllocatablePin(role="mcu", pin="13", gpio=6, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "spi-miso")),
        AllocatablePin(role="mcu", pin="14", gpio=7, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "spi-mosi")),
        AllocatablePin(role="mcu", pin="3", capabilities=("gpio", "input", "output", "interrupt")),
        AllocatablePin(role="mcu", pin="15", capabilities=("gpio", "input", "output", "pwm", "interrupt")),
        AllocatablePin(role="mcu", pin="16", capabilities=("gpio", "input", "output", "pwm", "interrupt")),
        AllocatablePin(role="mcu", pin="17", capabilities=("gpio", "input", "output", "pwm", "interrupt")),
        AllocatablePin(role="mcu", pin="20", capabilities=("gpio", "input", "output", "pwm", "interrupt")),
    ),
    placement_constraints=(
        PlacementConstraint(kind="decoupling_proximity", role="local_decoupling", parameters={"anchor_role": "mcu", "max_mm": 3.0}),
    ),
    electrical_assertions=(
        Assertion(code="stm32g0_supply_3v3", message="STM32G030F6P6 VDD/VDDA is a 3.3 V rail within its 2.0–3.6 V operating range"),
        Assertion(code="stm32g0_swd_reset_access", message="SWDIO, SWCLK/BOOT0, NRST, VTREF, and ground reach the physical SWD header"),
        Assertion(code="stm32g0_boot_low", message="PA14/BOOT0 has a 10 kΩ pulldown and no default bootloader switch"),
    ),
    source_documents=(
        SourceDocument(url=_STM32G030F6P6_DATASHEET, title="STM32G030x6/x8 data sheet", revision="DS12991 Rev. 4", reviewed_date=_REVIEW_DATE, sections=("Figure 5 TSSOP20 pinout", "Tables 12–14 alternate functions", "Table 19 I/O limits", "Table 21 operating conditions")),
        SourceDocument(url="https://www.st.com/resource/en/application_note/an5096-getting-started-with-stm32g0-mcus-hardware-development-stmicroelectronics.pdf", title="STM32G0 hardware development", revision="AN5096 Rev. 4", reviewed_date=_REVIEW_DATE, sections=("Figure 5 reset", "Figure 9 decoupling", "boot modes", "debug")),
    ),
)


ATTINY1604_SSN_MINIMAL = RecipeDefinition(
    recipe="attiny1604-ssn-minimal@1",
    family="attiny1604",
    exact_part="ATTINY1604-SSN",
    default_for_family=True,
    maturity="production",
    protected_aliases=("ATtiny1604", "tinyAVR"),
    identity_aliases=("attiny1604-ssn", "MCU_Microchip_ATtiny:ATtiny1604-SS"),
    required_sheet_roles=("mcu",),
    parameter_defaults={
        "logic_voltage_v": 3.3,
        "maximum_clock_mhz": 10,
        "rstpincfg": "updi",
        "adc_expansion": "none",
        "gpio_expansion": "none",
        "consumer_input_max_v": 3.3,
    },
    allowed_parameters={
        "logic_voltage_v": (3.3,),
        "maximum_clock_mhz": (10,),
        "rstpincfg": ("updi",),
        "adc_expansion": ("none", "ads1115x2"),
        "gpio_expansion": ("none", "mcp23017"),
        "consumer_input_max_v": (3.3,),
    },
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="updi", direction="bidirectional", required=False),
        Port(name="uart_tx", direction="output", required=False),
        Port(name="uart_rx", direction="input", required=False),
        Port(name="i2c_sda", direction="bidirectional", required=False),
        Port(name="i2c_scl", direction="bidirectional", required=False),
        Port(name="spi_mosi", direction="output", required=False),
        Port(name="spi_miso", direction="input", required=False),
        Port(name="spi_sck", direction="output", required=False),
        Port(name="spi_cs0", direction="output", required=False),
        *(Port(name=f"adc_{index}", direction="input", required=False) for index in range(8)),
        *(Port(name=f"output_{index}", direction="output", required=False) for index in range(16)),
        *(Port(name=f"parallel_{index}", direction="output", required=False) for index in range(16)),
    ),
    fixed_port_pins={
        "uart_tx": "11",
        "uart_rx": "12",
        "i2c_sda": "12",
        "i2c_scl": "13",
        "spi_mosi": "11",
        "spi_miso": "12",
        "spi_sck": "13",
        "spi_cs0": "2",
    },
    internal_nets=("updi",),
    parts=(
        Group(role="mcu", reference_prefix="U", value="ATTINY1604-SSN", symbol="MCU_Microchip_ATtiny:ATtiny1604-SS", footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm", sheet_role="mcu", mpn="ATTINY1604-SSN", datasheet=_ATTINY1604_DATASHEET),
        Group(role="local_decoupling", reference_prefix="C", value="100nF", symbol=_PASSIVE_CAP[0], footprint=_PASSIVE_CAP[1], sheet_role="mcu"),
        Group(role="bulk_decoupling", reference_prefix="C", value="1uF", symbol=_PASSIVE_CAP[0], footprint=_PASSIVE_CAP[1], sheet_role="mcu"),
        Group(role="updi_pullup", reference_prefix="R", value="10k", symbol=_PASSIVE_RES[0], footprint=_PASSIVE_RES[1], sheet_role="mcu"),
        Group(role="updi_header", reference_prefix="J", value="UPDI", symbol=_PROGRAM_HEADER[0], footprint=_PROGRAM_HEADER[1], sheet_role="mcu"),
    ),
    pins=(
        Pin(role="mcu", pin="1", net="vdd"),
        Pin(role="mcu", pin="14", net="gnd"),
        Pin(role="mcu", pin="10", net="updi"),
        Pin(role="local_decoupling", pin="1", net="vdd"),
        Pin(role="local_decoupling", pin="2", net="gnd"),
        Pin(role="bulk_decoupling", pin="1", net="vdd"),
        Pin(role="bulk_decoupling", pin="2", net="gnd"),
        Pin(role="updi_pullup", pin="1", net="vdd"),
        Pin(role="updi_pullup", pin="2", net="updi"),
        Pin(role="updi_header", pin="1", net="vdd"),
        Pin(role="updi_header", pin="2", net="updi"),
        Pin(role="updi_header", pin="3", net="gnd"),
    ),
    no_connects=(NoConnect(role="updi_header", pin="4"),),
    allocatable_pins=(
        AllocatablePin(role="mcu", pin="2", gpio=4, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "spi-cs")),
        AllocatablePin(role="mcu", pin="3", gpio=5, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc")),
        AllocatablePin(role="mcu", pin="4", gpio=6, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc")),
        AllocatablePin(role="mcu", pin="5", gpio=7, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc")),
        AllocatablePin(role="mcu", pin="6", capabilities=("gpio", "input", "output", "pwm", "interrupt", "uart-rx")),
        AllocatablePin(role="mcu", pin="7", capabilities=("gpio", "input", "output", "pwm", "interrupt", "uart-tx")),
        AllocatablePin(role="mcu", pin="8", capabilities=("gpio", "input", "output", "pwm", "interrupt")),
        AllocatablePin(role="mcu", pin="9", capabilities=("gpio", "input", "output", "pwm", "interrupt")),
        AllocatablePin(role="mcu", pin="11", gpio=1, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "uart-tx", "spi-mosi")),
        AllocatablePin(role="mcu", pin="12", gpio=2, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "uart-rx", "spi-miso", "i2c-sda")),
        AllocatablePin(role="mcu", pin="13", gpio=3, capabilities=("gpio", "input", "output", "pwm", "interrupt", "adc", "spi-sclk", "i2c-scl")),
    ),
    placement_constraints=(
        PlacementConstraint(kind="decoupling_proximity", role="local_decoupling", parameters={"anchor_role": "mcu", "max_mm": 3.0}),
    ),
    electrical_assertions=(
        Assertion(code="attiny1604_supply_3v3", message="ATtiny1604 VDD is a 3.3 V rail and the configured clock is limited to 10 MHz"),
        Assertion(code="attiny1604_updi_access", message="PA0/UPDI reaches the physical header through an optional 10 kΩ pull-up with no pull-down"),
        Assertion(code="attiny1604_no_pa0_reset", message="UPDI remains the default RSTPINCFG; user reset needs the separately owned power-cycle composition"),
    ),
    source_documents=(
        SourceDocument(url=_ATTINY1604_DATASHEET, title="ATtiny804/806/807/1604/1606/1607 data sheet", revision="DS40002312A", reviewed_date=_REVIEW_DATE, sections=("Table 2-1 pinout", "Table 5-1 port multiplexing", "UPDI", "operating speed grades")),
        SourceDocument(url="https://support.microchip.com/s/article/AVR---Hardware-Design-Considerations-for-UPDI-pin", title="AVR UPDI hardware design considerations", revision="current", reviewed_date=_REVIEW_DATE, sections=("target-side UPDI pull-up at least 10 kΩ",)),
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
            mpn="X32258MSB4SI",
            symbol="crystal-8mhz-3225:X32258MSB4SI",
            footprint="crystal-8mhz-3225:CRYSTAL-SMD_4P-L3.2-W2.5-BL",
            datasheet="https://www.lcsc.com/datasheet/C2682774.pdf",
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

_CH32V003_DATASHEET = "https://www.wch-ic.com/download/file?id=359"


# CH32V003F4P6 is the 20-lead family member: unlike the SOP-8 J4M6, its
# package exposes the complete SPI peripheral and enough independent GPIO for
# display, sensor, and indicator compositions.  PD1 remains dedicated to the
# WCH one-wire serial debug interface; PD7 remains the reset input.  The
# remaining pads are only tagged with peripheral capabilities documented for
# that physical pad, rather than treating every GPIO as ADC/SPI/I2C/UART.
CH32V003F4P6_MINIMAL = RecipeDefinition(
    recipe="ch32v003f4p6-minimal@1",
    family="ch32v003",
    exact_part="CH32V003F4P6",
    # C5187096 is currently retail-dry; preserve exact support but leave the
    # family default to the separately sourced F4U6.
    default_for_family=False,
    maturity="production",
    protected_aliases=("CH32V003", "CH32V003F4P6"),
    identity_aliases=("ch32v003f4p6", "ch32v003f4p6:CH32V003F4P6"),
    required_sheet_roles=("mcu",),
    parameter_defaults={"logic_voltage_v": 3.3, "consumer_input_max_v": 3.3},
    allowed_parameters={
        "logic_voltage_v": (3.3,),
        "consumer_input_max_v": (3.3,),
    },
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="swio", direction="bidirectional", required=False),
        Port(name="uart_tx", direction="output", required=False),
        Port(name="uart_rx", direction="input", required=False),
        Port(name="i2c_scl", direction="bidirectional", required=False),
        Port(name="i2c_sda", direction="bidirectional", required=False),
        Port(name="spi_sck", direction="output", required=False),
        Port(name="spi_mosi", direction="output", required=False),
        Port(name="spi_miso", direction="input", required=False),
        Port(name="spi_cs0", direction="output", required=False),
        *(Port(name=f"adc_{index}", direction="input", required=False) for index in range(8)),
        *(Port(name=f"input_{index}", direction="input", required=False) for index in range(16)),
        *(Port(name=f"output_{index}", direction="output", required=False) for index in range(16)),
        *(Port(name=f"parallel_{index}", direction="output", required=False) for index in range(16)),
    ),
    fixed_port_pins={
        "uart_tx": "2",  # PD5 / USART1_TX
        "uart_rx": "3",  # PD6 / USART1_RX
        "i2c_sda": "11",  # PC1 / I2C1_SDA
        "i2c_scl": "12",  # PC2 / I2C1_SCL
        "spi_cs0": "11",  # PC1 / SPI1_NSS; mutually exclusive with I2C SDA
        "spi_sck": "15",  # PC5 / SPI1_SCK
        "spi_mosi": "16",  # PC6 / SPI1_MOSI
        "spi_miso": "17",  # PC7 / SPI1_MISO
    },
    internal_nets=("nrst", "swio"),
    parts=(
        Group(
            role="mcu",
            reference_prefix="U",
            value="CH32V003F4P6",
            symbol="ch32v003f4p6:CH32V003F4P6",
            footprint="ch32v003f4p6:TSSOP-20_L6.5-W4.4-P0.65-LS6.4-BL",
            sheet_role="mcu",
            mpn="CH32V003F4P6",
            datasheet=_CH32V003_DATASHEET,
        ),
        Group(role="local_decoupling", reference_prefix="C", value="100nF", symbol=_PASSIVE_CAP[0], footprint=_PASSIVE_CAP[1], sheet_role="mcu"),
        Group(role="reset_pullup", reference_prefix="R", value="10k", symbol=_PASSIVE_RES[0], footprint=_PASSIVE_RES[1], sheet_role="mcu"),
        Group(role="reset_button", reference_prefix="SW", value="RESET", symbol="kmr232ngulclfs:KMR232NGULCLFS", footprint="kmr232ngulclfs:KMR232NGULCLFS", sheet_role="mcu", mpn="KMR232NGULCLFS"),
        Group(role="program_header", reference_prefix="J", value="WCH-LINK", symbol=_PROGRAM_HEADER[0], footprint=_PROGRAM_HEADER[1], sheet_role="mcu"),
    ),
    pins=(
        Pin(role="mcu", pin="9", net="vdd"),  # VDD
        Pin(role="mcu", pin="7", net="gnd"),  # VSS
        Pin(role="mcu", pin="18", net="swio"),  # PD1 / SWIO
        Pin(role="mcu", pin="4", net="nrst"),  # PD7 / NRST
        Pin(role="local_decoupling", pin="1", net="vdd"),
        Pin(role="local_decoupling", pin="2", net="gnd"),
        Pin(role="reset_pullup", pin="1", net="vdd"),
        Pin(role="reset_pullup", pin="2", net="nrst"),
        Pin(role="reset_button", pin="1", net="nrst"),
        Pin(role="reset_button", pin="2", net="gnd"),
        Pin(role="program_header", pin="1", net="vdd"),
        Pin(role="program_header", pin="2", net="gnd"),
        Pin(role="program_header", pin="3", net="swio"),
        Pin(role="program_header", pin="4", net="nrst"),
    ),
    allocatable_pins=(
        AllocatablePin(role="mcu", pin="1", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PD4 / AIN7
        AllocatablePin(role="mcu", pin="2", capabilities=("gpio", "input", "output", "interrupt", "adc", "uart-tx")),  # PD5 / AIN5, USART1_TX
        AllocatablePin(role="mcu", pin="3", capabilities=("gpio", "input", "output", "interrupt", "adc", "uart-rx")),  # PD6 / AIN6, USART1_RX
        AllocatablePin(role="mcu", pin="5", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PA1 / AIN1
        AllocatablePin(role="mcu", pin="6", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PA2 / AIN0
        AllocatablePin(role="mcu", pin="8", capabilities=("gpio", "input", "output", "interrupt")),  # PD0
        AllocatablePin(role="mcu", pin="10", capabilities=("gpio", "input", "output", "interrupt")),  # PC0
        AllocatablePin(role="mcu", pin="11", capabilities=("gpio", "input", "output", "interrupt", "i2c-sda", "spi-cs")),  # PC1
        AllocatablePin(role="mcu", pin="12", capabilities=("gpio", "input", "output", "interrupt", "i2c-scl")),  # PC2
        AllocatablePin(role="mcu", pin="13", capabilities=("gpio", "input", "output", "interrupt")),  # PC3
        AllocatablePin(role="mcu", pin="14", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PC4 / AIN2
        AllocatablePin(role="mcu", pin="15", capabilities=("gpio", "input", "output", "interrupt", "spi-sclk")),  # PC5 / SPI1_SCK
        AllocatablePin(role="mcu", pin="16", capabilities=("gpio", "input", "output", "interrupt", "spi-mosi")),  # PC6 / SPI1_MOSI
        AllocatablePin(role="mcu", pin="17", capabilities=("gpio", "input", "output", "interrupt", "spi-miso")),  # PC7 / SPI1_MISO
        AllocatablePin(role="mcu", pin="19", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PD2 / AIN3
        AllocatablePin(role="mcu", pin="20", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PD3 / AIN4
    ),
    placement_constraints=(
        PlacementConstraint(kind="decoupling_proximity", role="local_decoupling", parameters={"anchor_role": "mcu", "max_mm": 3.0}),
    ),
    electrical_assertions=(
        Assertion(code="ch32v003f4p6_supply_3v3", message="CH32V003F4P6 VDD is supplied from the reviewed 3.3 V logic rail, within the datasheet 2.7–5.5 V operating range"),
        Assertion(code="ch32v003f4p6_sdi_access", message="PD1/SWIO, VDD, ground, and NRST reach the physical WCH-Link header"),
        Assertion(code="ch32v003f4p6_reset_default", message="PD7/NRST has a 10 kΩ pull-up and reset switch to ground"),
    ),
    source_documents=(
        SourceDocument(
            url=_CH32V003_DATASHEET,
            title="WCH CH32V003 data sheet",
            revision="V1.8",
            reviewed_date="2026-10-11",
            sections=(
                "Chapter 2 package and ordering information",
                "Table 2-1 package pinouts",
                "Table 2-2 pin definitions and alternate functions",
                "Section 3.3.9 electrical characteristics",
                "Table 3-17 I/O characteristics",
            ),
        ),
        SourceDocument(
            url="https://www.lcsc.com/product-detail/Microcontroller-Units-MCUs-MPUs-SOCs_WCH-Jiangsu-Qin-Heng-CH32V003F4P6_C5187096.html",
            title="LCSC CH32V003F4P6 product identity listing",
            revision="C5187096",
            reviewed_date="2026-10-11",
            sections=("CH32V003F4P6 orderable TSSOP-20 identity listing; retail availability is separately checked",),
        ),
    ),
)


CH32V003F4U6_MINIMAL = RecipeDefinition(
    recipe="ch32v003f4u6-minimal@1",
    family="ch32v003",
    exact_part="CH32V003F4U6",
    default_for_family=True,
    maturity="production",
    protected_aliases=("CH32V003", "CH32V003F4U6"),
    identity_aliases=("ch32v003f4u6", "ch32v003f4u6:CH32V003F4U6"),
    required_sheet_roles=("mcu",),
    parameter_defaults={"logic_voltage_v": 3.3, "consumer_input_max_v": 3.3},
    allowed_parameters={
        "logic_voltage_v": (3.3,),
        "consumer_input_max_v": (3.3,),
    },
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="swio", direction="bidirectional", required=False),
        Port(name="uart_tx", direction="output", required=False),
        Port(name="uart_rx", direction="input", required=False),
        Port(name="i2c_scl", direction="bidirectional", required=False),
        Port(name="i2c_sda", direction="bidirectional", required=False),
        Port(name="spi_sck", direction="output", required=False),
        Port(name="spi_mosi", direction="output", required=False),
        Port(name="spi_miso", direction="input", required=False),
        Port(name="spi_cs0", direction="output", required=False),
        *(Port(name=f"adc_{index}", direction="input", required=False) for index in range(8)),
        *(Port(name=f"input_{index}", direction="input", required=False) for index in range(16)),
        *(Port(name=f"output_{index}", direction="output", required=False) for index in range(16)),
        *(Port(name=f"parallel_{index}", direction="output", required=False) for index in range(16)),
    ),
    fixed_port_pins={
        "uart_tx": "19",  # PD5 / USART1_TX
        "uart_rx": "20",  # PD6 / USART1_RX
        "i2c_sda": "8",  # PC1 / I2C1_SDA
        "i2c_scl": "9",  # PC2 / I2C1_SCL
        "spi_cs0": "8",  # PC1 / SPI1_NSS; mutually exclusive with I2C SDA
        "spi_sck": "12",  # PC5 / SPI1_SCK
        "spi_mosi": "13",  # PC6 / SPI1_MOSI
        "spi_miso": "14",  # PC7 / SPI1_MISO
    },
    internal_nets=("nrst", "swio"),
    parts=(
        Group(
            role="mcu",
            reference_prefix="U",
            value="CH32V003F4U6",
            symbol="ch32v003f4u6:CH32V003F4U6",
            footprint="ch32v003f4u6:QFN-20_L3.0-W3.0-P0.40-TL-EP1.7",
            sheet_role="mcu",
            mpn="CH32V003F4U6",
            datasheet=_CH32V003_DATASHEET,
        ),
        Group(role="local_decoupling", reference_prefix="C", value="100nF", symbol=_PASSIVE_CAP[0], footprint=_PASSIVE_CAP[1], sheet_role="mcu"),
        Group(role="reset_pullup", reference_prefix="R", value="10k", symbol=_PASSIVE_RES[0], footprint=_PASSIVE_RES[1], sheet_role="mcu"),
        Group(role="reset_button", reference_prefix="SW", value="RESET", symbol="kmr232ngulclfs:KMR232NGULCLFS", footprint="kmr232ngulclfs:KMR232NGULCLFS", sheet_role="mcu", mpn="KMR232NGULCLFS"),
        Group(role="program_header", reference_prefix="J", value="WCH-LINK", symbol=_PROGRAM_HEADER[0], footprint=_PROGRAM_HEADER[1], sheet_role="mcu"),
    ),
    pins=(
        Pin(role="mcu", pin="6", net="vdd"),  # VDD
        Pin(role="mcu", pin="4", net="gnd"),  # VSS
        Pin(role="mcu", pin="21", net="gnd"),  # exposed VSS pad
        Pin(role="mcu", pin="15", net="swio"),  # PD1 / SWIO
        Pin(role="mcu", pin="1", net="nrst"),  # PD7 / NRST
        Pin(role="local_decoupling", pin="1", net="vdd"),
        Pin(role="local_decoupling", pin="2", net="gnd"),
        Pin(role="reset_pullup", pin="1", net="vdd"),
        Pin(role="reset_pullup", pin="2", net="nrst"),
        Pin(role="reset_button", pin="1", net="nrst"),
        Pin(role="reset_button", pin="2", net="gnd"),
        Pin(role="program_header", pin="1", net="vdd"),
        Pin(role="program_header", pin="2", net="gnd"),
        Pin(role="program_header", pin="3", net="swio"),
        Pin(role="program_header", pin="4", net="nrst"),
    ),
    allocatable_pins=(
        AllocatablePin(role="mcu", pin="2", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PA1 / AIN1
        AllocatablePin(role="mcu", pin="3", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PA2 / AIN0
        AllocatablePin(role="mcu", pin="5", capabilities=("gpio", "input", "output", "interrupt")),  # PD0
        AllocatablePin(role="mcu", pin="7", capabilities=("gpio", "input", "output", "interrupt")),  # PC0
        AllocatablePin(role="mcu", pin="8", capabilities=("gpio", "input", "output", "interrupt", "i2c-sda", "spi-cs")),  # PC1
        AllocatablePin(role="mcu", pin="9", capabilities=("gpio", "input", "output", "interrupt", "i2c-scl")),  # PC2
        AllocatablePin(role="mcu", pin="10", capabilities=("gpio", "input", "output", "interrupt")),  # PC3
        AllocatablePin(role="mcu", pin="11", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PC4 / AIN2
        AllocatablePin(role="mcu", pin="12", capabilities=("gpio", "input", "output", "interrupt", "spi-sclk")),  # PC5 / SPI1_SCK
        AllocatablePin(role="mcu", pin="13", capabilities=("gpio", "input", "output", "interrupt", "spi-mosi")),  # PC6 / SPI1_MOSI
        AllocatablePin(role="mcu", pin="14", capabilities=("gpio", "input", "output", "interrupt", "spi-miso")),  # PC7 / SPI1_MISO
        AllocatablePin(role="mcu", pin="16", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PD2 / AIN3
        AllocatablePin(role="mcu", pin="17", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PD3 / AIN4
        AllocatablePin(role="mcu", pin="18", capabilities=("gpio", "input", "output", "interrupt", "adc")),  # PD4 / AIN7
        AllocatablePin(role="mcu", pin="19", capabilities=("gpio", "input", "output", "interrupt", "adc", "uart-tx")),  # PD5 / AIN5, USART1_TX
        AllocatablePin(role="mcu", pin="20", capabilities=("gpio", "input", "output", "interrupt", "adc", "uart-rx")),  # PD6 / AIN6, USART1_RX
    ),
    placement_constraints=(
        PlacementConstraint(kind="decoupling_proximity", role="local_decoupling", parameters={"anchor_role": "mcu", "max_mm": 3.0}),
    ),
    electrical_assertions=(
        Assertion(code="ch32v003f4u6_supply_3v3", message="CH32V003F4U6 VDD is supplied from the reviewed 3.3 V logic rail, within the datasheet 2.7–5.5 V operating range"),
        Assertion(code="ch32v003f4u6_sdi_access", message="PD1/SWIO, VDD, both VSS contacts, and NRST reach the physical WCH-Link header"),
        Assertion(code="ch32v003f4u6_reset_default", message="PD7/NRST has a 10 kΩ pull-up and reset switch to ground"),
    ),
    source_documents=(
        SourceDocument(
            url=_CH32V003_DATASHEET,
            title="WCH CH32V003 data sheet",
            revision="V1.8",
            reviewed_date="2026-10-11",
            sections=(
                "Chapter 2 package and ordering information",
                "Table 2-1 package pinouts",
                "Table 2-2 pin definitions and alternate functions",
                "Section 3.3.9 electrical characteristics",
                "Table 3-17 I/O characteristics",
            ),
        ),
        SourceDocument(
            url="https://www.lcsc.com/product-detail/Microcontroller-Units-MCUs-MPUs-SOCs_WCH-Jiangsu-Qin-Heng-CH32V003F4U6_C5299908.html",
            title="LCSC CH32V003F4U6 product identity listing",
            revision="C5299908",
            reviewed_date="2026-10-11",
            sections=("CH32V003F4U6 orderable QFN-20 identity listing; retail availability is separately checked",),
        ),
    ),
)

WAVE_A_MCU_RECIPES = (
    ESP32_S3_MINI_1U_N4R2_MINIMAL,
    ESP32_S3_WROOM_1_MINIMAL,
    ESP32_WROOM_32E_MINIMAL,
    ESP32_C3_MINI_1_MINIMAL,
    ESP32_C3_MINI_1_H4X_MINIMAL,
    ESP32_S3_MINI_1U_N4R2_UART,
    ESP32_C3_MINI_1_H4X_UART,
    STM32G030F6P6_MINIMAL,
    STM32F103C8T6_MINIMAL,
    ATTINY402_UPDI_MINIMAL,
    ATTINY402_SSNR_UPDI_MINIMAL,
    ATTINY412_UPDI_MINIMAL,
    ATTINY1614_UPDI_MINIMAL,
    ATTINY1604_SSN_MINIMAL,
    CH32V003F4P6_MINIMAL,
    CH32V003F4U6_MINIMAL,
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
    for port, pin in zip(
        ("uart_tx", "uart_rx"), _NATIVE_USB_UART_PINS.get(definition.recipe, ())
    ):
        if port in bindings:
            pins.append(Pin(role="mcu", pin=pin, net=port))
        else:
            no_connects.append(NoConnect(role="mcu", pin=pin))
    if definition.fixed_port_pins is not None:
        allocated_pins = {allocation.pin for allocation in resolved.selection.pin_allocations}
        for port, pin in definition.fixed_port_pins.items():
            if port in bindings and pin not in allocated_pins:
                pins.append(Pin(role="mcu", pin=pin, net=port))
    if definition.recipe in {"stm32g030f6p6-minimal@1", "attiny1604-ssn-minimal@1"}:
        i2c_ports = {"i2c_scl", "i2c_sda"}
        bound_i2c = i2c_ports.intersection(bindings)
        ads_enabled = resolved.parameters.get("adc_expansion") == "ads1115x2"
        gpio_enabled = resolved.parameters.get("gpio_expansion") == "mcp23017"
        if bound_i2c and bound_i2c != i2c_ports:
            raise ValueError(
                f"recipe {definition.recipe} requires both i2c_scl and i2c_sda for a physical I2C bus"
            )
        if bound_i2c:
            i2c_nets = {"i2c_scl": "i2c_scl", "i2c_sda": "i2c_sda"}
        elif ads_enabled or gpio_enabled:
            i2c_nets = {
                "i2c_scl": "mcu_expansion_i2c_scl",
                "i2c_sda": "mcu_expansion_i2c_sda",
            }
            internal_nets.extend(i2c_nets.values())
            for port, net in i2c_nets.items():
                pins.append(Pin(role="mcu", pin=definition.fixed_port_pins[port], net=net))
        else:
            i2c_nets = {}
        for port, net in sorted(i2c_nets.items()):
            role = f"{port}_pullup"
            parts.append(Group(role=role, reference_prefix="R", value="4.7k", symbol=_PASSIVE_RES[0], footprint=_PASSIVE_RES[1], sheet_role="mcu"))
            pins.extend((Pin(role=role, pin="1", net="vdd"), Pin(role=role, pin="2", net=net)))
        if ads_enabled:
            adc_ports = [f"adc_{index}" for index in range(8)]
            if not all(port in bindings for port in adc_ports):
                raise ValueError(f"recipe {definition.recipe} ADS1115 expansion requires eight ADC endpoints")
            parts.extend((
                Group(role="ads1115", reference_prefix="U", quantity=2, value="ADS1115IDGSR", symbol="ads1115:ADS1115IDGSR", footprint="ads1115:MSOP-10_L3.0-W3.0-P0.50-LS5.0-BL", sheet_role="mcu", mpn="ADS1115IDGSR", datasheet="https://www.ti.com/lit/ds/symlink/ads1115.pdf"),
                Group(role="ads1115_decoupling", reference_prefix="C", quantity=2, value="100nF", symbol=_PASSIVE_CAP[0], footprint=_PASSIVE_CAP[1], sheet_role="mcu"),
            ))
            for index in range(2):
                pins.extend((
                    Pin(role="ads1115", index=index, pin="1", net="gnd" if index == 0 else "vdd"),
                    Pin(role="ads1115", index=index, pin="3", net="gnd"),
                    Pin(role="ads1115", index=index, pin="8", net="vdd"),
                    Pin(role="ads1115", index=index, pin="9", net=i2c_nets["i2c_sda"]),
                    Pin(role="ads1115", index=index, pin="10", net=i2c_nets["i2c_scl"]),
                    Pin(role="ads1115_decoupling", index=index, pin="1", net="vdd"),
                    Pin(role="ads1115_decoupling", index=index, pin="2", net="gnd"),
                ))
                no_connects.append(NoConnect(role="ads1115", index=index, pin="2"))
                for offset, pin in enumerate(("4", "5", "6", "7")):
                    pins.append(Pin(role="ads1115", index=index, pin=pin, net=adc_ports[index * 4 + offset]))
        if gpio_enabled:
            endpoint_nets: list[str | None] = []
            for index in range(16):
                output = bindings.get(f"output_{index}")
                parallel = bindings.get(f"parallel_{index}")
                if output and parallel and output != parallel:
                    raise ValueError(f"recipe {definition.recipe} has conflicting output endpoint {index}")
                endpoint_nets.append(output or parallel)
            if not any(endpoint_nets):
                raise ValueError(f"recipe {definition.recipe} MCP23017 expansion requires output endpoints")
            parts.extend((
                Group(role="mcp23017", reference_prefix="U", value="MCP23017-E/SO", symbol="Interface_Expansion:MCP23017_SO", footprint="Package_SO:SOIC-28W_7.5x17.9mm_P1.27mm", sheet_role="mcu", mpn="MCP23017-E/SO", datasheet="https://ww1.microchip.com/downloads/en/DeviceDoc/20001952C.pdf"),
                Group(role="mcp23017_decoupling", reference_prefix="C", value="100nF", symbol=_PASSIVE_CAP[0], footprint=_PASSIVE_CAP[1], sheet_role="mcu"),
                Group(role="mcp23017_reset_pullup", reference_prefix="R", value="10k", symbol=_PASSIVE_RES[0], footprint=_PASSIVE_RES[1], sheet_role="mcu"),
            ))
            pins.extend((
                Pin(role="mcp23017", pin="9", net="vdd"), Pin(role="mcp23017", pin="10", net="gnd"),
                Pin(role="mcp23017", pin="12", net=i2c_nets["i2c_scl"]), Pin(role="mcp23017", pin="13", net=i2c_nets["i2c_sda"]),
                Pin(role="mcp23017", pin="15", net="gnd"), Pin(role="mcp23017", pin="16", net="gnd"), Pin(role="mcp23017", pin="17", net="gnd"), Pin(role="mcp23017", pin="18", net="mcp23017_reset"),
                Pin(role="mcp23017_decoupling", pin="1", net="vdd"), Pin(role="mcp23017_decoupling", pin="2", net="gnd"),
                Pin(role="mcp23017_reset_pullup", pin="1", net="vdd"), Pin(role="mcp23017_reset_pullup", pin="2", net="mcp23017_reset"),
            ))
            internal_nets.append("mcp23017_reset")
            for index, net in enumerate(endpoint_nets):
                pin = str(21 + index) if index < 8 else str(index - 7)
                (pins.append(Pin(role="mcp23017", pin=pin, net=net)) if net else no_connects.append(NoConnect(role="mcp23017", pin=pin)))
            no_connects.extend((NoConnect(role="mcp23017", pin="11"), NoConnect(role="mcp23017", pin="14"), NoConnect(role="mcp23017", pin="19"), NoConnect(role="mcp23017", pin="20")))
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
        # C3 GPIO8 must be high for manual or automatic download recovery.
        # Share any allocated application conductor without claiming it twice.
        for pin in _DOWNLOAD_BOOT_HIGH_PINS[definition.recipe]:
            allocation = next(
                (row for row in resolved.selection.pin_allocations if row.pin == pin),
                None,
            )
            net = allocation.net if allocation else f"download_boot_high_{pin}"
            role = f"download_boot_high_pullup_{pin}"
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
    usb_pins = _NATIVE_USB_PINS.get(definition.recipe)
    if usb_pins is not None:
        bound_usb = {"usb_dm", "usb_dp"}.intersection(bindings)
        if bound_usb and bound_usb != {"usb_dm", "usb_dp"}:
            raise ValueError(
                f"recipe {definition.recipe} requires usb_dm and usb_dp together"
            )
        allocated_pins = {allocation.pin for allocation in resolved.selection.pin_allocations}
        if bound_usb:
            # ESP32-C3 and ESP32-S3 native D-/D+ use the MCU-side 22R pair.
            # The USB-C recipe suppresses its duplicate pair for these module
            # recipes; retain separate raw and connector-side nets here.
            internal_nets.extend(("usb_dm_raw", "usb_dp_raw"))
            parts.extend(
                (
                    Group(
                        role="usb_dm_series",
                        reference_prefix="R",
                        value="22R",
                        mpn="RC0603FR-0722RL",
                        symbol=_PASSIVE_RES[0],
                        footprint=_PASSIVE_RES[1],
                        sheet_role="mcu",
                    ),
                    Group(
                        role="usb_dp_series",
                        reference_prefix="R",
                        value="22R",
                        mpn="RC0603FR-0722RL",
                        symbol=_PASSIVE_RES[0],
                        footprint=_PASSIVE_RES[1],
                        sheet_role="mcu",
                    ),
                )
            )
            pins.extend(
                (
                    Pin(role="mcu", pin=usb_pins[0], net="usb_dm_raw"),
                    Pin(role="usb_dm_series", pin="1", net="usb_dm_raw"),
                    Pin(role="usb_dm_series", pin="2", net="usb_dm"),
                    Pin(role="mcu", pin=usb_pins[1], net="usb_dp_raw"),
                    Pin(role="usb_dp_series", pin="1", net="usb_dp_raw"),
                    Pin(role="usb_dp_series", pin="2", net="usb_dp"),
                )
            )
        else:
            for pin in usb_pins:
                if pin not in allocated_pins:
                    no_connects.append(NoConnect(role="mcu", pin=pin))
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
