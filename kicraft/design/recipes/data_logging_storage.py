"""Production nonvolatile-storage recipes for data-logger compositions."""
from __future__ import annotations

from .models import (
    RecipeComponentGroup as Group,
    RecipeDefinition,
    RecipeElectricalAssertion as Assertion,
    RecipePinSpec as Pin,
    RecipePlacementConstraint as Placement,
    RecipePort as Port,
    RecipeSourceDocument as Source,
)

_CAP = ("Device:C", "Capacitor_SMD:C_0603_1608Metric")
_RES = ("Device:R", "Resistor_SMD:R_0603_1608Metric")
_DATASHEET = "https://ww1.microchip.com/downloads/en/DeviceDoc/24AA256-24LC256-24FC256-Data-Sheet-20001203V.pdf"

MICROCHIP_24LC256_I2C_EEPROM = RecipeDefinition(
    recipe="24lc256-i2c-eeprom@1",
    family="i2c-eeprom-256k",
    exact_part="24LC256T-I/SN",
    default_for_family=True,
    maturity="production",
    protected_aliases=("i2c-eeprom-256k", "24LC256", "24LC256T-I/SN"),
    identity_aliases=("24LC256",),
    required_sheet_roles=("storage",),
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="sda", direction="bidirectional"),
        Port(name="scl", direction="input"),
        # The address is deliberately a physical strap, rather than an assumed
        # software setting. The default is 0b000, hence 7-bit I2C address 0x50.
        Port(name="a0", direction="input", required=False, default_tie="gnd"),
        Port(name="a1", direction="input", required=False, default_tie="gnd"),
        Port(name="a2", direction="input", required=False, default_tie="gnd"),
        # Ground is the documented hardware write-enable state; a high WP pin
        # must never be presented as writable logger storage.
        Port(name="wp", direction="input", required=False, default_tie="gnd"),
    ),
    parts=(
        Group(
            role="storage",
            reference_prefix="U",
            value="24LC256T-I/SN",
            symbol="Memory_EEPROM:24LC256",
            footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
            sheet_role="storage",
            mpn="24LC256T-I/SN",
            datasheet=_DATASHEET,
        ),
        Group(
            role="decoupling",
            reference_prefix="C",
            value="100nF",
            symbol=_CAP[0],
            footprint=_CAP[1],
            sheet_role="storage",
        ),
        Group(
            role="sda_pullup",
            reference_prefix="R",
            value="4.7k",
            symbol=_RES[0],
            footprint=_RES[1],
            sheet_role="storage",
        ),
        Group(
            role="scl_pullup",
            reference_prefix="R",
            value="4.7k",
            symbol=_RES[0],
            footprint=_RES[1],
            sheet_role="storage",
        ),
    ),
    pins=(
        Pin(role="storage", pin="1", net="a0"),
        Pin(role="storage", pin="2", net="a1"),
        Pin(role="storage", pin="3", net="a2"),
        Pin(role="storage", pin="4", net="gnd"),
        Pin(role="storage", pin="5", net="sda"),
        Pin(role="storage", pin="6", net="scl"),
        Pin(role="storage", pin="7", net="wp"),
        Pin(role="storage", pin="8", net="vdd"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
        Pin(role="sda_pullup", pin="1", net="vdd"),
        Pin(role="sda_pullup", pin="2", net="sda"),
        Pin(role="scl_pullup", pin="1", net="vdd"),
        Pin(role="scl_pullup", pin="2", net="scl"),
    ),
    placement_constraints=(
        Placement(kind="decoupling_proximity", role="decoupling", parameters={"max_mm": 3.0}),
    ),
    electrical_assertions=(
        Assertion(
            code="eeprom_i2c_pullups",
            message="SDA and SCL each have a local 4.7 kOhm pull-up to the EEPROM supply rail",
        ),
        Assertion(
            code="eeprom_write_enabled_default",
            message="WP is physically strapped to ground, enabling writes at the default address 0x50",
        ),
    ),
    source_documents=(
        Source(
            url=_DATASHEET,
            title="Microchip 24AA256/24LC256/24FC256 serial EEPROM data sheet",
            revision="DS20001203V",
            reviewed_date="2026-10-11",
            sections=(
                "Pin descriptions",
                "Device addressing",
                "Write protection",
                "Power-on reset and low-voltage behavior",
                "Typical application",
            ),
        ),
    ),
)

DATA_LOGGING_STORAGE_RECIPES = (MICROCHIP_24LC256_I2C_EEPROM,)
