"""RP2040 minimal recipe on the composable port and ownership contract."""

from .models import (
    RecipeComponentGroup as Group,
    RecipeDefinition,
    RecipeElectricalAssertion as Assertion,
    RecipeNoConnectSpec as NoConnect,
    RecipePinSpec as Pin,
    RecipePlacementConstraint as PlacementConstraint,
    RecipePort as Port,
    RecipeSourceDocument as SourceDocument,
)
from .rp2040_minimal import RP2040_CORE


_INTERNAL_NETS = (
    "QSPI_CS",
    "QSPI_SCLK",
    "QSPI_SD0",
    "QSPI_SD1",
    "QSPI_SD2",
    "QSPI_SD3",
    "XIN",
    "XOUT_RAW",
    "XOUT",
    "DVDD_1V1",
    "bootsel",
    "run",
    "swdio",
    "swclk",
)
_NATIVE_USB_PINS = ("46", "47")
_SWD_PINS = ("24", "25")
_UART_PINS = {"uart_tx": "2", "uart_rx": "3"}


RP2040_MINIMAL_V2: RecipeDefinition = RP2040_CORE.model_copy(
    update={
        "recipe": "rp2040-minimal@2",
        "family": "rp2040",
        "exact_part": "RP2040",
        "default_for_family": True,
        "maturity": "production",
        "identity_aliases": (
            "RP2040",
            "MCU_RaspberryPi:RP2040",
        ),
        "protected_aliases": ("Raspberry Pi RP2040",),
        "parameter_defaults": {
            "native_usb": True,
            "programming": "native_usb",
            "flash_mbit": 16,
            "logic_voltage_v": 3.3,
            "consumer_input_max_v": 3.3,
        },
        "allowed_parameters": {
            "native_usb": (True, False),
            "programming": ("native_usb", "swd"),
            "flash_mbit": (16,),
            "logic_voltage_v": (3.3,),
            "consumer_input_max_v": (3.3,),
        },
        "ports": (
            Port(name="vdd", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="usb_dm", direction="bidirectional", required=False),
            Port(name="usb_dp", direction="bidirectional", required=False),
            Port(name="uart_tx", direction="output", required=False),
            Port(name="uart_rx", direction="input", required=False),
            *(
                Port(name=f"gpio{index}", direction="bidirectional", required=False)
                for index in range(30)
            ),
        ),
        "internal_nets": _INTERNAL_NETS,
        "pins": tuple(
            pin.model_copy(
                update={
                    "net": (
                        "vdd"
                        if pin.net == "3V3"
                        else "gnd"
                        if pin.net == "GND"
                        else "usb_dm"
                        if pin.net == "USB_DM"
                        else "usb_dp"
                        if pin.net == "USB_DP"
                        else pin.net.lower()
                        if pin.net.startswith("GPIO")
                        else pin.net
                    )
                }
            )
            for pin in RP2040_CORE.pins
        ),
        "placement_constraints": (
            PlacementConstraint(
                kind="decoupling_proximity",
                role="io_decoupling",
                parameters={"anchor_role": "mcu", "max_mm": 3.0},
            ),
            PlacementConstraint(
                kind="crystal_proximity",
                role="crystal",
                parameters={"anchor_role": "mcu", "max_mm": 8.0},
            ),
        ),
        "electrical_assertions": (
            Assertion(
                code="rp2040_bootsel_access",
                message="QSPI chip select reaches ground only through the physical 1 kΩ BOOTSEL path",
            ),
            Assertion(
                code="rp2040_native_usb_programming",
                message="When native_usb is enabled, RP2040 USB D-/D+ reach one physical USB data connector.",
            ),
            Assertion(
                code="rp2040_swd_recovery",
                message="When native_usb is disabled, SWDIO, SWCLK, RUN, VTREF, and ground reach a physical SWD header.",
            ),
        ),
        "source_documents": (
            SourceDocument(
                url="https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf",
                title="RP2040 Datasheet",
                revision="build-date 2025-02-20",
                reviewed_date="2026-10-11",
                sections=("1.4.2 Pin Descriptions", "1.4.3 GPIO Functions", "2.3.3 Debug", "2.7 Boot Sequence", "2.9 Power Supplies", "4.2 UART"),
            ),
            SourceDocument(
                url="https://datasheets.raspberrypi.com/rp2040/hardware-design-with-rp2040.pdf",
                title="Hardware design with RP2040",
                revision="release 2, build-date 2026-08-20",
                reviewed_date="2026-10-11",
                sections=("2.1 Power", "2.2 Flash storage", "2.3 Crystal oscillator", "2.4.1 USB", "3.3.2 Debug – SWD"),
            ),
            SourceDocument(
                url="https://abracon.com/datasheets/ABM8-272-T3.pdf",
                title="ABM8-272-T3 crystal specifications and mechanical dimensions",
                revision="Drawing 456603 revision B, 2024-09-16",
                reviewed_date="2026-09-11",
                sections=("Key Electrical Specifications", "Mechanical Dimensions"),
            ),
        ),
    }
)


def expand_rp2040_minimal(resolved):
    """Materialize either native USB recovery or a physically complete SWD recovery path."""
    from .registry import expand_static_definition

    definition = RP2040_MINIMAL_V2
    bindings = resolved.selection.port_bindings
    native_usb = resolved.parameters["native_usb"]
    programming = resolved.parameters["programming"]
    bound_usb = {"usb_dm", "usb_dp"}.intersection(bindings)
    if bound_usb and bound_usb != {"usb_dm", "usb_dp"}:
        raise ValueError("recipe rp2040-minimal@2 requires usb_dm and usb_dp together")
    bound_uart = {"uart_tx", "uart_rx"}.intersection(bindings)
    if bound_uart and bound_uart != {"uart_tx", "uart_rx"}:
        raise ValueError(
            "recipe rp2040-minimal@2 UART bridge mode requires uart_tx and uart_rx together"
        )
    if bound_uart and {"gpio0", "gpio1"}.intersection(bindings):
        raise ValueError(
            "recipe rp2040-minimal@2 bridge mode reserves GPIO0/GPIO1 for UART0"
        )
    if bound_uart:
        definition = definition.model_copy(
            update={
                "pins": (
                    *(
                        pin for pin in definition.pins
                        if not (
                            (pin.role == "mcu" and pin.pin in _UART_PINS.values())
                            or (pin.role == "cast_left" and pin.index in {2, 3})
                        )
                    ),
                    *(Pin(role="mcu", pin=pin, net=port) for port, pin in _UART_PINS.items()),
                ),
                "no_connects": (
                    *definition.no_connects,
                    NoConnect(role="cast_left", index=2, pin="1"),
                    NoConnect(role="cast_left", index=3, pin="1"),
                ),
            }
        )
    if native_usb:
        if programming != "native_usb" or bound_usb != {"usb_dm", "usb_dp"}:
            raise ValueError(
                "recipe rp2040-minimal@2 native USB requires programming='native_usb' and a complete USB pair"
            )
        # Raspberry Pi hardware guide §2.4.1 requires 27-ohm termination at
        # the MCU, not the generic USB receptacle's 22-ohm pair.
        raw_usb = dict(zip(_NATIVE_USB_PINS, ("usb_dm_raw", "usb_dp_raw")))
        definition = definition.model_copy(update={
            "internal_nets": (*definition.internal_nets, *raw_usb.values()),
            "parts": (
                *definition.parts,
                *(Group(
                    role=f"{port}_series", reference_prefix="R", value="27R",
                    symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric",
                    sheet_role="mcu",
                ) for port in ("usb_dm", "usb_dp")),
            ),
            "pins": (
                *(pin.model_copy(update={"net": raw_usb[pin.pin]})
                  if pin.role == "mcu" and pin.pin in raw_usb else pin
                  for pin in definition.pins),
                *(pin for port in ("usb_dm", "usb_dp") for pin in (
                    Pin(role=f"{port}_series", pin="1", net=f"{port}_raw"),
                    Pin(role=f"{port}_series", pin="2", net=port),
                )),
            ),
        })
        return expand_static_definition(definition, resolved)
    if programming != "swd":
        raise ValueError(
            "recipe rp2040-minimal@2 native_usb=False requires programming='swd'"
        )
    if bound_usb:
        raise ValueError(
            "recipe rp2040-minimal@2 native_usb=False cannot bind the RP2040 USB PHY"
        )

    pins = [
        pin
        for pin in definition.pins
        if not (
            (pin.role == "mcu" and pin.pin in _NATIVE_USB_PINS)
            or (pin.role == "cast_right" and pin.index in {15, 16})
        )
    ]
    pins.extend(
        (
            Pin(role="mcu", pin="24", net="swclk"),
            Pin(role="mcu", pin="25", net="swdio"),
            Pin(role="swd_header", pin="1", net="vdd"),
            Pin(role="swd_header", pin="2", net="gnd"),
            Pin(role="swd_header", pin="3", net="swdio"),
            Pin(role="swd_header", pin="4", net="swclk"),
            Pin(role="swd_header", pin="5", net="run"),
        )
    )
    no_connects = [
        pin
        for pin in definition.no_connects
        if not (pin.role == "mcu" and pin.pin in _SWD_PINS)
    ]
    no_connects.extend(
        (
            NoConnect(role="mcu", pin="46"),
            NoConnect(role="mcu", pin="47"),
            NoConnect(role="cast_right", index=15, pin="1"),
            NoConnect(role="cast_right", index=16, pin="1"),
            NoConnect(role="swd_header", pin="6"),
        )
    )
    return expand_static_definition(
        definition.model_copy(
            update={
                "parts": (
                    *definition.parts,
                    Group(
                        role="swd_header",
                        reference_prefix="J",
                        value="SWD",
                        symbol="Connector_Generic:Conn_01x06",
                        footprint="Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical",
                        sheet_role="mcu",
                    ),
                ),
                "pins": tuple(pins),
                "no_connects": tuple(no_connects),
            }
        ),
        resolved,
    )
