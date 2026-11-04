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
    parts[3].footprint = "usblc6-2sc6:SOT-23-6_L2.9-W1.6-P0.95-LS2.8-BL"
    parts[3].mpn = "USBLC6-2SC6"

    def conn(net, *pins):
        return NetConnection(
            net_name=net,
            sheet="MCU",
            endpoints=[PinEndpoint(ref=ref, pin=pin) for ref, pin in pins],
        )

    # Misleading net names on purpose: physical pins must decide the verdict.
    connections = [
        conn("ALPHA", ("U1", "23"), ("R3", "1")),
        conn("BETA", ("R3", "2"), ("U2", "6")),
        conn("GAMMA", ("U2", "1"), ("J1", "A7"), ("J1", "B7")),
        conn("DELTA", ("U1", "24"), ("R4", "1")),
        conn("EPSILON", ("R4", "2"), ("U2", "4")),
        conn("ZETA", ("U2", "3"), ("J1", "A6"), ("J1", "B6")),
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
        ] if c.net_name == "GAMMA" else [("U2", "3"), ("J1", "A7"), ("J1", "B7")])
        for c in connections
    ]
    bad = check_mcu_programming_access(_bom(*parts, connections=swapped))
    assert not bad.ok
    assert any(o.startswith("native_usb_programming_required:U1") for o in bad.offenders)


def test_native_usb_rejects_legacy_internally_shorted_feedthrough():
    from kicraft.design.models import NetConnection, PinEndpoint

    esd = _part("U2", "USBLC6-2SC6", "usblc6-2sc6:USBLC6-2SC6_C2687116")
    esd.footprint = "usblc6-2sc6:SOT-23-6_L2.9-W1.6-P0.95-LS2.8-BL"
    esd.mpn = "USBLC6-2SC6"
    parts = [
        _part("U1", "ESP32-S3-MINI-1-N8", "esp32-s3-mini-1:ESP32-S3-MINI-1-N8"),
        esd, USB, _part("SW1", "BOOT button"), _part("SW2", "RESET (EN) button"),
    ]
    def conn(net, *pins):
        return NetConnection(net_name=net, sheet="MCU", endpoints=[
            PinEndpoint(ref=ref, pin=pin) for ref, pin in pins
        ])
    connections = [
        conn("DM", ("U1", "23"), ("U2", "3")),
        conn("DP", ("U1", "24"), ("U2", "4")),
        conn("DM_CONN", ("U2", "1"), ("J1", "A7"), ("J1", "B7")),
        conn("DP_CONN", ("U2", "6"), ("J1", "A6"), ("J1", "B6")),
    ]
    result = check_mcu_programming_access(_bom(*parts, connections=connections))
    assert not result.ok
    assert any("shorted" in offender for offender in result.offenders)


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


def _uart_fixture(monkeypatch, model):
    import kicraft.design.synthesis.validation as validation

    if model == "s3":
        name = "ESP32-S3-MINI-1U-N4R2"
        footprint = "esp32-s3-mini-1u-n4r2:BULETM-SMD_ESPRESSIF_ESP32-S3-MINI-1U-N8"
        module_pins = ("3", "1", "39", "40", "45", "4")
    else:
        name = "ESP32-C3-MINI-1-H4X"
        footprint = "esp32-c3-mini-1-h4x:WIFIM-SMD_ESP32-C3-MINI-1"
        module_pins = ("3", "1", "31", "30", "8", "23")
    mcu = _part("U1", name, symbol=f"{name.lower()}:{name}")
    mcu.mpn, mcu.footprint = name, footprint
    header = _part("J2", "UART PROGRAM", symbol="Connector_Generic:Conn_01x06")
    supply = _part("J3", "3V3 INPUT", symbol="Connector_Generic:Conn_01x02")
    cap = _part("C1", "100n", symbol="Device:C")
    parts = [mcu, header, supply, cap]
    info = {
        "U1": {n: {"name": "", "type": ""} for n in module_pins},
        "J2": {n: {"name": n, "type": "passive"} for n in "123456"},
        "J3": {n: {"name": n, "type": "passive"} for n in "12"},
        "C1": {n: {"name": n, "type": "passive"} for n in "12"},
    }
    monkeypatch.setattr(validation, "_pin_info_by_ref", lambda bom: (info, {}))

    def endpoint(ref, pin):
        return SimpleNamespace(ref=ref, pin=pin)

    connections = []
    for n, (net, module_pin) in enumerate(zip(
        ("+3V3", "GND", "TX", "RX", "EN", "BOOT"), module_pins
    ), 1):
        endpoints = [endpoint("U1", module_pin), endpoint("J2", str(n))]
        if n <= 2:
            endpoints.extend((endpoint("J3", str(n)), endpoint("C1", str(n))))
        connections.append(SimpleNamespace(net_name=net, endpoints=endpoints))
    return _bom(*parts, connections=connections), info


def test_reviewed_rom_uart_requires_complete_powered_physical_path(monkeypatch):
    from copy import deepcopy

    for model in ("s3", "c3"):
        bom, info = _uart_fixture(monkeypatch, model)
        assert check_mcu_programming_access(bom).ok
        assert mcu_programming_facts(bom)["rom_uart_proven"] == ["U1"]
        for pin in "123456":
            broken = deepcopy(bom)
            for connection in broken.connections:
                connection.endpoints = [
                    ep for ep in connection.endpoints
                    if (ep.ref, ep.pin) != ("J2", pin)
                ]
            assert not check_mcu_programming_access(broken).ok, (model, pin)

        for ref in ("J3", "C1"):
            broken = deepcopy(bom)
            broken.parts = [p for p in broken.parts if p.ref != ref]
            assert not check_mcu_programming_access(broken).ok, (model, ref)

        broken = deepcopy(bom)
        broken.connections[2].endpoints[1].pin = "4"
        broken.connections[3].endpoints[1].pin = "3"
        assert not check_mcu_programming_access(broken).ok

        broken = deepcopy(bom)
        broken.connections[3].net_name = broken.connections[2].net_name
        assert not check_mcu_programming_access(broken).ok

        broken = deepcopy(bom)
        broken.no_connect_pins = [SimpleNamespace(ref="J2", pin="5")]
        assert not check_mcu_programming_access(broken).ok

        del info["J2"]["6"]
        assert not check_mcu_programming_access(bom).ok


def test_rom_uart_inventory_is_only_provisional_before_wiring(monkeypatch):
    bom, _ = _uart_fixture(monkeypatch, "s3")
    bom.connections = []
    result = check_mcu_programming_access(bom)
    assert result.ok and "unproven" in result.message
    facts = mcu_programming_facts(bom)
    assert not facts["wiring_committed"]
    assert facts["rom_uart_proven"] == []


def test_rom_uart_rejects_unreviewed_identity_and_unpowered_rails(monkeypatch):
    from copy import deepcopy

    bom, info = _uart_fixture(monkeypatch, "c3")
    unreviewed = deepcopy(bom)
    unreviewed.parts[0].mpn = "ESP32-C3-MINI-1-other"
    assert not check_mcu_programming_access(unreviewed).ok

    mislabeled = deepcopy(bom)
    mislabeled.parts[1].value = "UART PROGRAM EXTRA"
    assert not check_mcu_programming_access(mislabeled).ok

    no_supply = deepcopy(bom)
    for connection in no_supply.connections:
        connection.endpoints = [ep for ep in connection.endpoints if ep.ref != "J3"]
    assert not check_mcu_programming_access(no_supply).ok

    no_bypass = deepcopy(bom)
    no_bypass.connections[1].endpoints = [
        ep for ep in no_bypass.connections[1].endpoints if ep.ref != "C1"
    ]
    assert not check_mcu_programming_access(no_bypass).ok

    info["J3"].pop("2")
    assert not check_mcu_programming_access(bom).ok


def test_rom_uart_accepts_reviewed_powered_converter_not_floating_output(monkeypatch):
    from copy import deepcopy

    for model in ("s3", "c3"):
        bom, info = _uart_fixture(monkeypatch, model)
        converter = _part(
            "U2", "K7803M-1000R3",
            symbol="ylptec-k7803m-1000r3:K7803M-1000R3_C22749779",
        )
        converter.mpn = "K7803M-1000R3"
        converter.footprint = "ylptec-k7803m-1000r3:PWRM-TH_K78XXM-1000R3"
        bom.parts.append(converter)
        info["U2"] = {
            "1": {"name": "Vin", "type": "unspecified"},
            "2": {"name": "GND", "type": "unspecified"},
            "3": {"name": "Vo", "type": "unspecified"},
        }
        for ref, mpn in (("C2", "CL31B106KBHNNNE"), ("C3", "EMK316BB7226ML-T")):
            capacitor = _part(ref, mpn, symbol="Device:C")
            capacitor.mpn = mpn
            bom.parts.append(capacitor)
            info[ref] = {n: {"name": n, "type": "passive"} for n in "12"}
        # J3 is now the external high-voltage input, not a 3V3 source.
        for connection in bom.connections:
            connection.endpoints = [
                ep for ep in connection.endpoints if (ep.ref, ep.pin) != ("J3", "1")
            ]
        bom.connections[0].endpoints.append(SimpleNamespace(ref="U2", pin="3"))
        bom.connections[1].endpoints.append(SimpleNamespace(ref="U2", pin="2"))
        bom.connections[0].endpoints.append(SimpleNamespace(ref="C3", pin="1"))
        bom.connections[1].endpoints.extend((
            SimpleNamespace(ref="C2", pin="2"), SimpleNamespace(ref="C3", pin="2"),
        ))
        bom.connections.append(SimpleNamespace(
            net_name="VIN18", endpoints=[
                SimpleNamespace(ref="U2", pin="1"), SimpleNamespace(ref="J3", pin="1"),
                SimpleNamespace(ref="C2", pin="1"),
            ],
        ))
        assert check_mcu_programming_access(bom).ok
        for ref, pin in (("U2", "1"), ("U2", "2"), ("U2", "3"),
                         ("J3", "1"), ("J3", "2"), ("C2", "1"), ("C3", "1")):
            broken = deepcopy(bom)
            for connection in broken.connections:
                connection.endpoints = [
                    ep for ep in connection.endpoints if (ep.ref, ep.pin) != (ref, pin)
                ]
            assert not check_mcu_programming_access(broken).ok, (model, ref, pin)
        for pin in "123456":
            broken = deepcopy(bom)
            for connection in broken.connections:
                connection.endpoints = [
                    ep for ep in connection.endpoints if (ep.ref, ep.pin) != ("J2", pin)
                ]
            assert not check_mcu_programming_access(broken).ok, (model, pin)

        unknown = deepcopy(bom)
        unreviewed = next(part for part in unknown.parts if part.ref == "U2")
        unreviewed.mpn = "UNREVIEWED-CONVERTER"
        unreviewed.symbol = "unknown:unreviewed"
        info["U2"]["3"]["type"] = "power_out"
        assert not check_mcu_programming_access(unknown).ok


def test_two_real_uart_recipes_cannot_power_each_other_via_program_headers():
    from collections import defaultdict

    from kicraft.design.models import RecipeSelection
    from kicraft.design.recipes.registry import expand_selections
    from kicraft.design.synthesis.validation import _rom_uart_programming_proven

    selections = [
        RecipeSelection(
            recipe=recipe, instance=f"core{index}", sheets={"mcu": "MCU"},
            port_bindings={"vdd": "+3V3", "gnd": "GND"},
        )
        for index, recipe in enumerate((
            "esp32-s3-mini-1u-n4r2-uart-minimal@1",
            "esp32-c3-mini-1-h4x-uart-minimal@1",
        ))
    ]
    expansions = expand_selections(selections)
    connections = defaultdict(list)
    for expansion in expansions:
        for connection in expansion.connections:
            connections[connection.net_name].extend(connection.endpoints)
    bom = _bom(
        *(part for expansion in expansions for part in expansion.parts),
        connections=[
            SimpleNamespace(net_name=net, endpoints=endpoints)
            for net, endpoints in connections.items()
        ],
    )
    bom.no_connect_pins = [
        pin for expansion in expansions for pin in expansion.no_connect_pins
    ]
    mcus = [part for part in bom.parts if part.recipe_role == "mcu"]
    assert len(mcus) == 2
    assert not any(_rom_uart_programming_proven(bom, mcu) for mcu in mcus)

    supply = _part("J99", "EXTERNAL 3V3", symbol="Connector_Generic:Conn_01x02")
    bom.parts.append(supply)
    for connection in bom.connections:
        if connection.net_name in {"+3V3", "GND"}:
            connection.endpoints.append(SimpleNamespace(
                ref="J99", pin="1" if connection.net_name == "+3V3" else "2",
            ))
    assert all(_rom_uart_programming_proven(bom, mcu) for mcu in mcus)
