"""Fix 3b: deterministic edge-zone fallback for edge-mount connectors."""
from __future__ import annotations

from kicraft.design.synthesis.autoplacer import (
    DEFAULT_EDGE_CONNECTOR_ZONE,
    _edge_connector_zone_injections,
)


def test_unzoned_usb_c_receptacle_gets_edge_zone():
    out = _edge_connector_zone_injections(
        [("J1", "Connector_USB:USB_C_Receptacle_USB2.0_Type-C-GCT-USB4105")], {}
    )
    assert out == {"J1": {"edge": DEFAULT_EDGE_CONNECTOR_ZONE}}


def test_vendored_easyeda_usb_c_gets_edge_zone():
    # The exact footprint that floated on the esp32-led-matrix board.
    out = _edge_connector_zone_injections([("J1", "USB-C_SMD-TYPE-C-31-M-12")], {})
    assert out == {"J1": {"edge": DEFAULT_EDGE_CONNECTOR_ZONE}}


def test_barrel_jack_gets_edge_zone():
    out = _edge_connector_zone_injections(
        [("J3", "Connector_BarrelJack:BarrelJack_Horizontal")], {}
    )
    assert out == {"J3": {"edge": DEFAULT_EDGE_CONNECTOR_ZONE}}


def test_screw_terminal_gets_edge_zone():
    # KC-YJ7Q69: the facing gate only covers zoned refs, so a screw terminal
    # must never depend on the BOM stage remembering to zone it.
    out = _edge_connector_zone_injections(
        [
            ("J2", "screw-terminal-5mm-3p:CONN-TH_3P-P5.00_WJ126V-5.0-3P"),
            ("J4", "TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-2_1x02_P5.00mm_Horizontal"),
        ],
        {},
    )
    assert out == {
        "J2": {"edge": DEFAULT_EDGE_CONNECTOR_ZONE},
        "J4": {"edge": DEFAULT_EDGE_CONNECTOR_ZONE},
    }


def test_vendored_dc_barrel_jack_gets_edge_zone():
    out = _edge_connector_zone_injections(
        [("J1", "dc-barrel-jack-5-5-2-1:DC-IN-TH_DC005-5.5-2.1")], {}
    )
    assert out == {"J1": {"edge": DEFAULT_EDGE_CONNECTOR_ZONE}}


def test_already_zoned_connector_is_left_untouched():
    out = _edge_connector_zone_injections(
        [("J1", "USB-C_SMD-TYPE-C-31-M-12")],
        {"J1": {"edge": "left"}},  # LLM already chose an edge
    )
    assert out == {}, "must not override an existing component_zone"


def test_non_edge_footprints_are_not_zoned():
    out = _edge_connector_zone_injections(
        [
            ("R1", "Resistor_SMD:R_0402_1005Metric"),
            ("U1", "Package_QFP:LQFP-48_7x7mm_P0.5mm"),
            ("J2", "Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical"),
        ],
        {},
    )
    # internal pin header is not an off-board edge connector -> no injection
    assert out == {}


def test_multiple_parts_mixed():
    out = _edge_connector_zone_injections(
        [
            ("J1", "USB-C_SMD-TYPE-C-31-M-12"),
            ("R1", "Resistor_SMD:R_0402_1005Metric"),
            ("J5", "Connector_USB:USB_A_Vertical"),
        ],
        {},
    )
    assert out == {
        "J1": {"edge": DEFAULT_EDGE_CONNECTOR_ZONE},
        "J5": {"edge": DEFAULT_EDGE_CONNECTOR_ZONE},
    }
