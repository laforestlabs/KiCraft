"""The architecture stage decides the MCU programming interface up front so an
ESP32-class design never parks at the wiring stage on a flashing question (which
would force a BOM re-run). Default: prefer a native-USB ESP32 variant (flash over
USB, no bridge); fall back to an onboard CH340C USB-UART bridge only for the classic
ESP32-WROOM-32, which has no native USB."""

from __future__ import annotations

from kicraft.server.stage_contracts import build_stage_response_contract
from kicraft.server.stage_prompts import build_system as _build_system


def build_system(stage: str) -> str:
    state = {"architecture": {"sheets": [{"name": "POWER"}]}} if stage == "bom" else {}
    return _build_system(build_stage_response_contract(stage, state))


def test_architecture_spec_decides_programming_interface_early():
    low = build_system("architecture").lower()
    assert "programming interface" in low  # decided here, not discovered at wiring
    # Default recommendation: a native-USB ESP32 variant, flashed over USB, no bridge.
    assert "native" in low and "usb" in low
    assert "esp32-s3" in low  # the vendored native-USB default
    # CH340C is the fallback only for the classic ESP32 (no native USB).
    assert "ch340c" in low
    assert "auto-reset" in low


def test_wiring_spec_connects_a_provided_interface_instead_of_asking():
    low = build_system("wiring").lower()
    # When the BOM already carries a programming interface, wiring must connect it.
    assert "must connect it" in low
    assert "ch340c" in low


def test_bom_spec_requires_a_recovery_mechanism_beyond_usb():
    # §9.29: native USB omits the BRIDGE, never the recovery mechanism. The
    # BOM prompt must demand BOOT + EN/RESET (or strap test pads / another
    # approved mechanism) and call out that a bare USB connector is
    # insufficient for first download and recovery (KC-7FVTPW defect #1).
    low = build_system("bom").lower()
    assert "no usb-uart bridge" in low  # native USB means no BRIDGE ...
    assert "boot" in low  # ... but still BOOT + EN/RESET ...
    assert "en/reset" in low
    assert "test pads" in low  # ... or labeled strap test pads
    assert "usb connector alone is not sufficient" in low
    assert "first download and recovery" in low
    # classic ESP32 keeps the bridge + auto-reset network
    assert "ch340c" in low
    assert "dtr/rts auto-reset" in low
    # Recovery support is represented as explicit component groups.
    assert "component groups" in low
