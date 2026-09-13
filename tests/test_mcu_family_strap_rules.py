"""§9.29 family strap/reset rules (self-eval 2026-07-19 run_10 / run_30).

"Has a USB connector" is not a workable programming story for bootloader-strap
families: an RP2040 without BOOTSEL/SWD cannot re-enter its ROM bootloader
after first flash, and an ESP32 without BOOT+EN buttons (or DTR/RTS auto-reset
via a USB-UART bridge) cannot be put into download mode. Both shipped through
§9.29 and were observer-gated at cap 50.
"""
from __future__ import annotations

from types import SimpleNamespace

from kicraft.design.synthesis.validation import (
    check_mcu_programming_access,
    mcu_programming_facts,
)


def _part(ref, value, symbol="Device:R", sourcing_note=None):
    return SimpleNamespace(
        ref=ref, value=value, symbol=symbol, footprint="Resistor_SMD:R_0402",
        sourcing_note=sourcing_note, mpn=None,
    )


def _bom(*parts, connections=()):
    return SimpleNamespace(
        parts=list(parts), connections=list(connections), no_connect_pins=[]
    )


# A real, resolvable USB-C socket part (exposes DN1/DN2 and DP1/DP2).
USB = _part("J1", "TYPE-C-31-M-12", symbol="usb-c-16p:TYPE-C-31-M-12")
# A header whose value merely names the MCU is NOT a controller or a socket.
MCU_HEADER = _part("J2", "ESP32-S3 UART PROGRAM", symbol="Connector_Generic:Conn_01x06")


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
    # Families outside the strap rules keep the existing contract:
    # any programming-access part (here USB) satisfies part presence.
    r = check_mcu_programming_access(_bom(_part("U1", "nRF52840 module"), USB))
    assert r.ok, r.offenders


# --- STM32 BOOT0 rule (2026-07-27 fix-plan P2.4, self-eval run_24) ----------
#
# STM32's ROM bootloader (USB-DFU and UART alike) is only entered with BOOT0
# HIGH at reset. run_24 shipped an STM32F042 whose assumed programming path
# was native-USB DFU with no BOOT0 access part anywhere -- and died in
# reconcile asking for exactly the strap this rule now demands at BOM commit.

def test_stm32_usb_only_fails():
    r = check_mcu_programming_access(_bom(_part("U1", "STM32F042F6P6"), USB))
    assert not r.ok
    assert any("BOOT0" in o for o in r.offenders)


def test_stm32_with_boot0_test_pad_passes():
    r = check_mcu_programming_access(
        _bom(_part("U1", "STM32F042F6P6"), USB, _part("TP1", "BOOT0 pad"))
    )
    assert r.ok, r.offenders


def test_stm32_with_swd_header_passes():
    r = check_mcu_programming_access(
        _bom(_part("U1", "STM32F103C8T6"), USB, _part("J2", "SWD debug header"))
    )
    assert r.ok, r.offenders


def test_stm32_with_boot_button_passes():
    r = check_mcu_programming_access(
        _bom(_part("U1", "STM32F042F6P6"), USB, _part("SW1", "BOOT0 button"))
    )
    assert r.ok, r.offenders


def test_stm32_with_usb_uart_bridge_passes():
    r = check_mcu_programming_access(
        _bom(_part("U1", "STM32F103C8T6"), USB,
             _part("U2", "CH340C USB-UART bridge"))
    )
    assert r.ok, r.offenders


# --- native-USB programming circuits (2026-09-12 board-quality contracts) ---
#
# The reviewed factory-native families (ESP32-S3 MINI/WROOM, ESP32-C3 MINI,
# RP2040) program over their own USB pair: §9.29 must see a physical data
# socket, and a header or power-only connector cannot stand in for it.

def test_native_family_without_usb_socket_fails_at_bom_commit():
    r = check_mcu_programming_access(
        _bom(
            _part("U1", "ESP32-S3-WROOM-1"),
            MCU_HEADER,
            _part("SW1", "BOOT button"),
            _part("SW2", "RESET (EN) button"),
        )
    )
    assert not r.ok
    assert any(o.startswith("native_usb_programming_required:U1") for o in r.offenders)


def test_header_named_usb_does_not_satisfy_native_programming():
    r = check_mcu_programming_access(
        _bom(
            _part("U1", "RP2040", symbol="MCU_RaspberryPi:RP2040"),
            _part("J3", "USB header", symbol="Connector_Generic:Conn_01x04"),
            _part("SW1", "BOOTSEL button"),
        )
    )
    assert not r.ok
    assert any(o.startswith("native_usb_programming_required:U1") for o in r.offenders)


def test_native_family_with_data_socket_passes_presence():
    r = check_mcu_programming_access(
        _bom(
            _part("U1", "ESP32-S3-MINI-1"),
            USB,
            _part("SW1", "BOOT button"),
            _part("SW2", "RESET (EN) button"),
        )
    )
    assert r.ok, r.offenders


def test_mcu_programming_facts_list_mcu_not_header_named_after_it():
    bom = _bom(
        _part("U4", "ESP32-S3-MINI-1-N8", symbol="esp32-s3-mini-1:ESP32-S3-MINI-1-N8"),
        MCU_HEADER,
        USB,
        _part("SW1", "BOOT button"),
        _part("SW2", "RESET (EN) button"),
    )
    facts = mcu_programming_facts(bom)
    assert any("U4" in row for row in facts["mcus"])
    assert not any("J2" in row for row in facts["mcus"])


def test_native_usb_wired_network_passes_and_renamed_nets_cannot_bypass():
    from kicraft.design.models import NetConnection, PinEndpoint

    def part(ref, value, symbol):
        return _part(ref, value, symbol=symbol)

    parts = [
        part("U1", "ESP32-S3-MINI-1-N8", "esp32-s3-mini-1:ESP32-S3-MINI-1-N8"),
        part("R3", "22R", "Device:R"),
        part("R4", "22R", "Device:R"),
        part("U2", "USBLC6-2SC6", "usblc6-2sc6:USBLC6-2SC6_C2687116"),
        USB,
        _part("SW1", "BOOT button"),
        _part("SW2", "RESET (EN) button"),
    ]

    def conn(net, *pins):
        return NetConnection(
            net_name=net,
            sheet="MCU",
            endpoints=[PinEndpoint(ref=ref, pin=pin) for ref, pin in pins],
        )

    # Misleading net names on purpose: physical pins must decide the verdict.
    connections = [
        conn("ALPHA", ("U1", "23"), ("R3", "1")),
        conn("BETA", ("R3", "2"), ("U2", "3")),
        conn("GAMMA", ("U2", "1"), ("J1", "A7"), ("J1", "B7")),
        conn("DELTA", ("U1", "24"), ("R4", "1")),
        conn("EPSILON", ("R4", "2"), ("U2", "4")),
        conn("ZETA", ("U2", "6"), ("J1", "A6"), ("J1", "B6")),
        conn("GND", ("U2", "2"), ("J1", "A1B12")),
        conn("VBUS", ("J1", "A4B9"), ("U2", "5")),
    ]
    r = check_mcu_programming_access(_bom(*parts, connections=connections))
    assert r.ok, r.offenders

    # Swapping the connector's D-/D+ pins breaks the physical proof.
    swapped = [
        c
        if c.net_name not in {"GAMMA", "ZETA"}
        else conn(c.net_name, *[
            ("U2", "1"), ("J1", "A6"), ("J1", "B6")
        ] if c.net_name == "GAMMA" else [("U2", "6"), ("J1", "A7"), ("J1", "B7")])
        for c in connections
    ]
    bad = check_mcu_programming_access(_bom(*parts, connections=swapped))
    assert not bad.ok
    assert any(o.startswith("native_usb_programming_required:U1") for o in bad.offenders)


def test_native_family_with_only_a_uart_bridge_fails():
    """A USB-UART bridge is a real programming path for the classic ESP32, but
    it is not the native-family data connector."""
    r = check_mcu_programming_access(
        _bom(
            _part("U1", "ESP32-S3-MINI-1"),
            _part("J2", "CH340C USB-UART bridge", symbol="ch340c:CH340C"),
            _part("SW1", "BOOT button"),
            _part("SW2", "RESET (EN) button"),
        )
    )
    assert not r.ok
    assert any(o.startswith("native_usb_programming_required:U1") for o in r.offenders)


def test_native_usb_open_data_line_fails_the_wired_proof():
    """A physically valid connector cannot cover an open D-/D+ line."""
    from kicraft.design.models import NetConnection, PinEndpoint

    parts = [
        _part("U1", "ESP32-S3-MINI-1-N8", symbol="esp32-s3-mini-1:ESP32-S3-MINI-1-N8"),
        _part("R3", "22R"),
        _part("R4", "22R"),
        _part("U2", "USBLC6-2SC6", symbol="usblc6-2sc6:USBLC6-2SC6_C2687116"),
        USB,
        _part("SW1", "BOOT button"),
        _part("SW2", "RESET (EN) button"),
    ]

    def conn(net, *pins):
        return NetConnection(
            net_name=net,
            sheet="MCU",
            endpoints=[PinEndpoint(ref=ref, pin=pin) for ref, pin in pins],
        )

    # D- path is complete; D+ series resistor R4 is wired on its MCU side only.
    connections = [
        conn("ALPHA", ("U1", "23"), ("R3", "1")),
        conn("BETA", ("R3", "2"), ("U2", "3")),
        conn("GAMMA", ("U2", "1"), ("J1", "A7"), ("J1", "B7")),
        conn("DELTA", ("U1", "24"), ("R4", "1")),
        conn("GND", ("U2", "2"), ("J1", "A1B12")),
        conn("VBUS", ("J1", "A4B9"), ("U2", "5")),
    ]
    r = check_mcu_programming_access(_bom(*parts, connections=connections))
    assert not r.ok
    assert any(o.startswith("native_usb_programming_required:U1") for o in r.offenders)
