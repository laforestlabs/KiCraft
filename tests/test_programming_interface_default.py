"""The architecture stage decides the MCU programming interface up front so an
ESP32-class design never parks at the wiring stage on a flashing question (which
would force a BOM re-run). Default: prefer a native-USB ESP32 variant (flash over
USB, no bridge); fall back to an onboard CH340C USB-UART bridge only for the classic
ESP32-WROOM-32, which has no native USB."""
from __future__ import annotations

from kicraft.server.stage_driver import build_system


def test_architecture_spec_decides_programming_interface_early():
    low = build_system("architecture").lower()
    assert "programming interface" in low    # decided here, not discovered at wiring
    # Default recommendation: a native-USB ESP32 variant, flashed over USB, no bridge.
    assert "native" in low and "usb" in low
    assert "esp32-s3" in low                  # the vendored native-USB default
    # CH340C is the fallback only for the classic ESP32 (no native USB).
    assert "ch340c" in low
    assert "auto-reset" in low


def test_wiring_spec_connects_a_provided_interface_instead_of_asking():
    low = build_system("wiring").lower()
    # When the BOM already carries a programming interface, wiring must connect it.
    assert "must connect it" in low
    assert "ch340c" in low
