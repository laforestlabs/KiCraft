"""RP2040 minimal recipe on the composable port and ownership contract."""

from .models import (
    RecipeDefinition,
    RecipeElectricalAssertion as Assertion,
    RecipePlacementConstraint as PlacementConstraint,
    RecipePort as Port,
    RecipeSourceDocument as SourceDocument,
)
from .rp2040_minimal import RP2040_MINIMAL


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
)


RP2040_MINIMAL_V2: RecipeDefinition = RP2040_MINIMAL.model_copy(
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
        "ports": (
            Port(name="vdd", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="usb_dm", direction="bidirectional"),
            Port(name="usb_dp", direction="bidirectional"),
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
            for pin in RP2040_MINIMAL.pins
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
                message="QSPI chip select has a physical BOOTSEL path",
            ),
            Assertion(
                code="rp2040_native_usb_programming",
                message=(
                    "Native USB D-/D+ reach one physical USB data connector, "
                    "with BOOTSEL-to-QSPI_CS first-flash and recovery access"
                ),
            ),
        ),
        "source_documents": (
            SourceDocument(
                url="https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf",
                title="RP2040 Datasheet",
                revision="build-date 2025-02-20",
                reviewed_date="2026-09-09",
                sections=("2.1 Pinout", "5.2.3 USB", "5.2.4 QSPI"),
            ),
            SourceDocument(
                url="https://datasheets.raspberrypi.com/rp2040/hardware-design-with-rp2040.pdf",
                title="Hardware design with RP2040",
                revision="release 2, build-date 2026-08-20",
                reviewed_date="2026-09-11",
                sections=("2.1 Power", "2.2 Flash storage", "2.3 Crystal oscillator", "2.4.1 USB"),
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
