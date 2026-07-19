"""§9.29 family strap/reset rules (self-eval 2026-07-19 run_10 / run_30).

"Has a USB connector" is not a workable programming story for bootloader-strap
families: an RP2040 without BOOTSEL/SWD cannot re-enter its ROM bootloader
after first flash, and an ESP32 without BOOT+EN buttons (or DTR/RTS auto-reset
via a USB-UART bridge) cannot be put into download mode. Both shipped through
§9.29 and were observer-gated at cap 50.
"""
from __future__ import annotations

from types import SimpleNamespace

from kicraft.design.synthesis.validation import check_mcu_programming_access


def _part(ref, value, symbol="Device:R", sourcing_note=None):
    return SimpleNamespace(
        ref=ref, value=value, symbol=symbol, footprint="Resistor_SMD:R_0402",
        sourcing_note=sourcing_note, mpn=None,
    )


def _bom(*parts):
    return SimpleNamespace(
        parts=list(parts), connections=[], no_connect_pins=[]
    )


USB = _part("J1", "USB-C receptacle")


def test_rp2040_usb_only_fails():
    r = check_mcu_programming_access(_bom(_part("U1", "RP2040"), USB))
    assert not r.ok
    assert any("BOOTSEL" in o for o in r.offenders)


def test_rp2040_with_bootsel_button_passes():
    r = check_mcu_programming_access(
        _bom(_part("U1", "RP2040"), USB, _part("SW1", "BOOTSEL button"))
    )
    assert r.ok, r.offenders


def test_rp2040_with_swd_header_passes():
    r = check_mcu_programming_access(
        _bom(_part("U1", "RP2040"), USB, _part("J2", "SWD header 1x4"))
    )
    assert r.ok, r.offenders


def test_esp32_usb_only_fails():
    r = check_mcu_programming_access(_bom(_part("U1", "ESP32-C3-MINI-1"), USB))
    assert not r.ok
    assert any("download mode" in o for o in r.offenders)


def test_esp32_with_boot_and_reset_buttons_passes():
    r = check_mcu_programming_access(
        _bom(
            _part("U1", "ESP32-S3-WROOM-1"), USB,
            _part("SW1", "BOOT button"), _part("SW2", "RESET (EN) button"),
        )
    )
    assert r.ok, r.offenders


def test_esp32_boot_button_alone_fails():
    r = check_mcu_programming_access(
        _bom(_part("U1", "ESP32-S3-WROOM-1"), USB, _part("SW1", "BOOT button"))
    )
    assert not r.ok


def test_esp32_with_usb_uart_bridge_passes():
    r = check_mcu_programming_access(
        _bom(_part("U1", "ESP32-WROOM-32"), USB,
             _part("U2", "CH340C USB-UART bridge"))
    )
    assert r.ok, r.offenders


def test_esp32_with_strap_test_pads_passes():
    r = check_mcu_programming_access(
        _bom(_part("U1", "ESP32-C3-MINI-1"), USB,
             _part("TP1", "IO9 strap pad"), _part("TP2", "EN pad"))
    )
    assert r.ok, r.offenders


def test_generic_mcu_with_usb_unchanged():
    # Families outside the two strap rules keep the existing contract:
    # any programming-access part (here USB) satisfies part presence.
    r = check_mcu_programming_access(_bom(_part("U1", "STM32F103C8T6"), USB))
    assert r.ok, r.offenders
