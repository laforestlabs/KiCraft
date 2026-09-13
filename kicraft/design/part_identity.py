"""Reviewed, directional device/order-code and family membership relations.

This is not a substitute for package, sourcing, or electrical validation. A
qualified request never widens to its base device or another package. Only case
and surrounding whitespace are insignificant; punctuation in order codes is
preserved. Unknown identities have equality, not inferred prefix/suffix aliases.
"""

from __future__ import annotations


# Manufacturer evidence reviewed 2026-09-11. These are explicit memberships,
# not interchangeable-device claims and not rules for decoding other suffixes.
# ADI MAX485 ordering table: MAX485ESA+ and MAX485ESA+T, 8-lead narrow SO;
# +T changes the shipping carrier, not the package or device.
# https://www.analog.com/en/products/max485.html
# Nordic nRF52840 Product Specification, Ordering information: QIAA is the
# aQFN-73 variant, R7 is the 7-inch reel (not a radio module).
# https://docs.nordicsemi.com/r/bundle/ps_nrf52840/page/ordering_info.html
# TI ULN2003A orderable device: ULN2003ADR, D/SOIC-16, large tape-and-reel.
# ULN2003 is the brief's unqualified device-series designation, not ULN2004.
# https://www.ti.com/product/ULN2003A/part-details/ULN2003ADR
# Microchip DS21952A, Product Identification System (p. 45), example (b):
# MCP23017-E/SO, extended-temperature 28-lead SOIC (not SPI MCP23S17).
# https://ww1.microchip.com/downloads/en/DeviceDoc/21952a.pdf
_DEVICE_MEMBERS: dict[str, frozenset[str]] = {
    "max485": frozenset({"max485esa+", "max485esa+t"}),
    "max485esa+": frozenset({"max485esa+t"}),
    "nrf52840": frozenset({"nrf52840-qiaa-r7"}),
    # TI's device is the ULN2003A; "ULN2003" is the brief's unqualified series
    # designation (see the sourcing comment above). A group naming either the
    # series or the A-variant is the same reviewed device, so both own the
    # requirement; the curated bundle still supplies the orderable identity.
    "uln2003": frozenset({"uln2003adr", "uln2003a"}),
    "uln2003a": frozenset({"uln2003a", "uln2003adr"}),
    "mcp23017": frozenset({"mcp23017-e/so"}),
    # Espressif ESP32-S3-WROOM-1 datasheet v1.8, Table 1-1 "Series Comparison":
    # N8R8 and N16R8 are the same module — 18.0 x 25.5 x 3.1 mm package, the
    # same pin map, the same -40 ~ 65 C ambient grade and the same 8 MB
    # Octal-SPI PSRAM — differing only in Quad-SPI flash capacity (8 vs 16 MB).
    # Neither is the other's base device or a different package, so serving one
    # with the other is a memory-capacity deviation the BOM must LEDGER
    # (bom.substitutions, §9.33); it must never be a silent substitution.
    # https://documentation.espressif.com/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf
    "esp32-s3-wroom-1-n16r8": frozenset({"esp32-s3-wroom-1-n8r8"}),
    "esp32-s3-wroom-1-n8r8": frozenset({"esp32-s3-wroom-1-n16r8"}),
}


# ST's STM32L031K6 product page lists STM32L031K6T6 in the STM32L0 series;
# ordering scheme: K=32 pins, 6=32-Kbyte flash, T=LQFP, 6=-40..85 Celsius.
# No inference is made for other L0 devices, densities, or package codes.
# https://www.st.com/en/microcontrollers-microprocessors/stm32l031k6.html
# https://www.st.com/resource/en/datasheet/stm32l031k4.pdf
_FAMILY_MEMBERS: dict[str, frozenset[str]] = {
    "stm32l0": frozenset({"stm32l031k6t6"}),
}


def is_part_family(identity: str) -> bool:
    """Whether this exact identity is a reviewed non-device family selector."""
    return identity.strip().casefold() in _FAMILY_MEMBERS


def matches_part_identity(requested: str, candidate: str) -> bool:
    """Whether candidate satisfies requested identity, without broadening it.

    Family self-equality is deliberately false: a family can own a typed
    architecture requirement, but cannot serve as that requirement's hardware.
    Callers must prefer an explicitly declared MPN over display values/labels.
    """
    requested_key = requested.strip().casefold()
    candidate_key = candidate.strip().casefold()
    if not requested_key or not candidate_key:
        return False
    family_members = _FAMILY_MEMBERS.get(requested_key)
    if family_members is not None:
        return candidate_key in family_members
    if requested_key == candidate_key:
        return True
    return candidate_key in _DEVICE_MEMBERS.get(requested_key, ())
