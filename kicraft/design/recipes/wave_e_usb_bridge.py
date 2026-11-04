"""Complete USB-C UART bridge recipes whose silicon owns the USB PHY."""
from __future__ import annotations

from .models import RecipeComponentGroup as Group, RecipeDefinition, RecipeElectricalAssertion as Assertion, RecipeNoConnectSpec as NoConnect, RecipePinSpec as Pin, RecipePlacementConstraint as Placement, RecipePort as Port, RecipeSourceDocument as Source


def _part(role, prefix, value, symbol, footprint, *, quantity=1, mpn=None):
    return Group(role=role, reference_prefix=prefix, value=value, symbol=symbol, footprint=footprint, sheet_role="interface", quantity=quantity, mpn=mpn)


def _passive(role, prefix, value, quantity=1):
    return _part(role, prefix, value, "Device:C" if prefix == "C" else "Device:R", "Capacitor_SMD:C_0603_1608Metric" if prefix == "C" else "Resistor_SMD:R_0603_1608Metric", quantity=quantity)


CP2102N_USB_UART = RecipeDefinition(
    recipe="cp2102n-usb-uart@1", family="usb-uart-bridge", exact_part="CP2102N",
    maturity="production", protected_aliases=("usb-uart-bridge", "CP2102N", "CP2102N USB-UART bridge"), identity_aliases=("CP2102N", "CP2102N USB-UART bridge"), required_sheet_roles=("interface",),
    parameter_defaults={"logic_voltage": 3.3}, allowed_parameters={"logic_voltage": (3.3,)},
    ports=(Port(name="vbus", direction="power"), Port(name="vdd", direction="power"), Port(name="gnd", direction="power"), Port(name="usb_dp", direction="bidirectional", required=False), Port(name="usb_dm", direction="bidirectional", required=False), Port(name="tx", direction="output"), Port(name="rx", direction="input"), Port(name="dtr_n", direction="output", required=False), Port(name="rts_n", direction="output", required=False)),
    internal_nets=("usb_dp", "usb_dm", "cc1", "cc2", "dp_conn", "dm_conn", "shield", "reset", "vbus_sense"),
    parts=(
        _part("bridge", "U", "CP2102N-A02-GQFN28", "Interface_USB:CP2102N-Axx-xQFN28", "Package_DFN_QFN:QFN-28-1EP_5x5mm_P0.5mm_EP3.35x3.35mm", mpn="CP2102N-A02-GQFN28"),
        _part("connector", "J", "TYPE-C-31-M-12", "usb-c-16p:TYPE-C-31-M-12", "usb-c-16p:USB-C_SMD-TYPE-C-31-M-12_1", mpn="TYPE-C-31-M-12"),
        _part("esd", "U", "USBLC6-2SC6", "st-usblc6-2sc6:USBLC6-2SC6", "st-usblc6-2sc6:SOT-23-6_L2.9-W1.6-P0.95-LS2.8-BL", mpn="USBLC6-2SC6"),
        _passive("cc_pulldown", "R", "5.1k", 2),
        _passive("vdd_decap", "C", "4.7uF"),
        _passive("vdd_high_frequency_decap", "C", "100nF"),
        _passive("regin_decap", "C", "4.7uF"),
        _passive("regin_high_frequency_decap", "C", "100nF"),
        _passive("reset_pullup", "R", "1k"),
        _passive("vbus_sense_upper", "R", "22.1k"),
        _passive("vbus_sense_lower", "R", "47.5k"),
        _part("program_header", "J", "UART-PROG", "Connector_Generic:Conn_01x06", "Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical"),
    ),
    pins=(
        # Datasheet figures 2.3 and 2.6: externally supplied 3.3 V bypasses
        # the internal regulator; USB VBUS is sensed through its divider.
        Pin(role="bridge", pin="3", net="gnd"), Pin(role="bridge", pin="4", net="usb_dp"), Pin(role="bridge", pin="5", net="usb_dm"), Pin(role="bridge", pin="6", net="vdd"), Pin(role="bridge", pin="7", net="vdd"), Pin(role="bridge", pin="8", net="vbus_sense"), Pin(role="bridge", pin="9", net="reset"), Pin(role="bridge", pin="24", net="rts_n"), Pin(role="bridge", pin="25", net="rx"), Pin(role="bridge", pin="26", net="tx"), Pin(role="bridge", pin="28", net="dtr_n"), Pin(role="bridge", pin="29", net="gnd"),
        Pin(role="connector", pin="A4B9", net="vbus"), Pin(role="connector", pin="B4A9", net="vbus"), Pin(role="connector", pin="A1B12", net="gnd"), Pin(role="connector", pin="B1A12", net="gnd"), Pin(role="connector", pin="A5", net="cc1"), Pin(role="connector", pin="B5", net="cc2"), Pin(role="connector", pin="A6", net="dp_conn"), Pin(role="connector", pin="B6", net="dp_conn"), Pin(role="connector", pin="A7", net="dm_conn"), Pin(role="connector", pin="B7", net="dm_conn"), *(Pin(role="connector", pin=str(pin), net="shield") for pin in range(1, 5)),
        Pin(role="cc_pulldown", index=0, pin="1", net="cc1"), Pin(role="cc_pulldown", index=0, pin="2", net="gnd"), Pin(role="cc_pulldown", index=1, pin="1", net="cc2"), Pin(role="cc_pulldown", index=1, pin="2", net="gnd"),
        Pin(role="esd", pin="1", net="dm_conn"), Pin(role="esd", pin="3", net="dp_conn"), Pin(role="esd", pin="2", net="gnd"), Pin(role="esd", pin="5", net="vbus"), Pin(role="esd", pin="6", net="usb_dm"), Pin(role="esd", pin="4", net="usb_dp"),
        Pin(role="vdd_decap", pin="1", net="vdd"), Pin(role="vdd_decap", pin="2", net="gnd"),
        Pin(role="vdd_high_frequency_decap", pin="1", net="vdd"), Pin(role="vdd_high_frequency_decap", pin="2", net="gnd"),
        Pin(role="regin_decap", pin="1", net="vdd"), Pin(role="regin_decap", pin="2", net="gnd"),
        Pin(role="regin_high_frequency_decap", pin="1", net="vdd"), Pin(role="regin_high_frequency_decap", pin="2", net="gnd"),
        Pin(role="reset_pullup", pin="1", net="vdd"), Pin(role="reset_pullup", pin="2", net="reset"),
        Pin(role="vbus_sense_upper", pin="1", net="vbus"), Pin(role="vbus_sense_upper", pin="2", net="vbus_sense"),
        Pin(role="vbus_sense_lower", pin="1", net="vbus_sense"), Pin(role="vbus_sense_lower", pin="2", net="gnd"),
        Pin(role="program_header", pin="1", net="gnd"), Pin(role="program_header", pin="2", net="vdd"), Pin(role="program_header", pin="3", net="tx"), Pin(role="program_header", pin="4", net="rx"), Pin(role="program_header", pin="5", net="dtr_n"), Pin(role="program_header", pin="6", net="rts_n"),
    ),
    no_connects=(NoConnect(role="connector", pin="A8"), NoConnect(role="connector", pin="B8"), *(NoConnect(role="bridge", pin=str(pin)) for pin in ("1", "2", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "27"))),
    placement_constraints=(Placement(kind="decoupling_proximity", role="bridge", parameters={"max_mm": 3.0}),),
    electrical_assertions=(Assertion(code="cp2102n_usb_c_uart", message="CP2102N owns one protected USB-C D+/D- pair. VDD and VREGIN share the external 3.3 V UART supply; a 22.1 kΩ/47.5 kΩ divider senses USB VBUS. USB compliance remains unmeasured."),),
    source_documents=(Source(url="https://www.silabs.com/documents/public/data-sheets/cp2102n-datasheet.pdf", title="Silicon Labs CP2102N", revision="1.5", reviewed_date="2026-10-11", sections=("5.1 QFN28 Pin Definitions", "2.1 Power, Figure 2.3 regulator bypass", "2.3 USB, Figure 2.6 VBUS divider")),),
)


def expand_cp2102n_usb_uart(resolved):
    """Leave optional modem-control contacts explicitly unconnected when unused."""
    from .registry import expand_static_definition

    optional = {
        "dtr_n": (("bridge", "28"), ("program_header", "5")),
        "rts_n": (("bridge", "24"), ("program_header", "6")),
    }
    absent = tuple(
        pin
        for port, pins in optional.items()
        if port not in resolved.selection.port_bindings
        for pin in pins
    )
    definition = CP2102N_USB_UART
    if absent:
        absent_pins = set(absent)
        definition = definition.model_copy(
            update={
                "pins": tuple(
                    pin for pin in definition.pins
                    if (pin.role, pin.pin) not in absent_pins
                ),
                "no_connects": (
                    *definition.no_connects,
                    *(NoConnect(role=role, pin=pin) for role, pin in absent),
                ),
            }
        )
    return expand_static_definition(definition, resolved)

WAVE_E_USB_BRIDGE_RECIPES = (CP2102N_USB_UART,)
