"""Production display and external-LED circuit recipes.

Defaults are intentionally finite: the four-digit display is four common-cathode
CA56-12EWA digits driven by one MAX7219; the matrix interface is a 1/32-scan
HUB75 output.  Firmware scan rate, optical brightness, external-strip current,
and heatsink performance remain board-integration/physical-validation concerns.
"""
from __future__ import annotations

from .models import (
    RecipeComponentGroup as Group,
    RecipeDefinition,
    RecipeElectricalAssertion as Assertion,
    RecipeNoConnectSpec as NoConnect,
    RecipePinSpec as Pin,
    RecipePlacementConstraint as Placement,
    RecipePort as Port,
    RecipeSourceDocument as Source,
)

_C = ("Device:C", "Capacitor_SMD:C_0603_1608Metric")
_R = ("Device:R", "Resistor_SMD:R_0603_1608Metric")


def _part(role: str, prefix: str, value: str, symbol: str, footprint: str, *, quantity: int = 1, mpn: str | None = None) -> Group:
    return Group(role=role, reference_prefix=prefix, value=value, symbol=symbol, footprint=footprint, sheet_role="interface", quantity=quantity, mpn=mpn)


def _passive(role: str, prefix: str, value: str, quantity: int = 1) -> Group:
    symbol, footprint = _C if prefix == "C" else _R
    return _part(role, prefix, value, symbol, footprint, quantity=quantity)


def _recipe(*, recipe: str, family: str, exact: str, ports: tuple[Port, ...], internal: tuple[str, ...], parts: tuple[Group, ...], pins: tuple[Pin, ...], source: Source, assertions: tuple[Assertion, ...], no_connects: tuple[NoConnect, ...] = (), parameters: dict | None = None, allowed: dict | None = None, aliases: tuple[str, ...] = (), safety: bool = False, default: bool = False) -> RecipeDefinition:
    constraints = [Placement(kind="decoupling_proximity", role=parts[0].role, parameters={"max_mm": 3.0})]
    if safety:
        constraints.append(Placement(kind="thermal_current_path", role=parts[0].role, parameters={"independent_review_required": True, "copper_area_required": True}))
    return RecipeDefinition(recipe=recipe, family=family, exact_part=exact, default_for_family=default, maturity="production", protected_aliases=(family, exact, *aliases), identity_aliases=(exact, *aliases), required_sheet_roles=("interface",), parameter_defaults=parameters or {}, allowed_parameters=allowed or {}, ports=ports, internal_nets=internal, parts=parts, pins=pins, no_connects=no_connects, placement_constraints=tuple(constraints), electrical_assertions=assertions, source_documents=(source,))


# A board-side three-contact connector is deliberately used here rather than an
# onboard WS2812: the external strip is the requested load.  The finite default
# is translated through a 5 V SN74HCT245 and has a 330 ohm source terminator
# plus 100 nF local bypass at the connector. The host supply budgets strip load.
WS2812_EXTERNAL_STRIP = _recipe(
    recipe="ws2812-external-strip-interface@1", family="ws2812-external-strip",
    exact="WS2812-EXTERNAL-STRIP",
    aliases=("WS2812 addressable LED strip",),
    ports=(Port(name="vdd_5v", direction="power"), Port(name="gnd", direction="power"), Port(name="data_in", direction="input")),
    internal=("data_shifted", "data_series"),
    parts=(
        _part("connector", "J", "WS2812 STRIP", "Connector_Generic:Conn_01x03", "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"),
        _part("level_shifter", "U", "SN74HCT245PWR-JSM", "sn74hct245:SN74HCT245PWR-JSM", "sn74hct245:TSSOP-20_L6.5-W4.4-P0.65-LS6.4-BL", mpn="SN74HCT245PWR-JSM"),
        _passive("series_resistor", "R", "330R"), _passive("bulk", "C", "100nF"),
    ),
    pins=(
        Pin(role="connector", pin="1", net="vdd_5v"), Pin(role="connector", pin="2", net="data_series"), Pin(role="connector", pin="3", net="gnd"),
        Pin(role="level_shifter", pin="20", net="vdd_5v"), Pin(role="level_shifter", pin="10", net="gnd"), Pin(role="level_shifter", pin="1", net="vdd_5v"), Pin(role="level_shifter", pin="19", net="gnd"), Pin(role="level_shifter", pin="2", net="data_in"), Pin(role="level_shifter", pin="18", net="data_shifted"),
        *(Pin(role="level_shifter", pin=str(pin), net="gnd") for pin in range(3, 10)),
        Pin(role="series_resistor", pin="1", net="data_shifted"), Pin(role="series_resistor", pin="2", net="data_series"), Pin(role="bulk", pin="1", net="vdd_5v"), Pin(role="bulk", pin="2", net="gnd"),
    ),
    no_connects=tuple(NoConnect(role="level_shifter", pin=str(pin)) for pin in range(11, 18)),
    source=Source(url="https://datasheet.lcsc.com/datasheet/pdf/bc8264a6d62c958a89e25ec8cc82b690.pdf?productCode=C2761795", title="Worldsemi WS2812B V5", revision="V5", reviewed_date="2026-10-11", sections=("Typical Application Circuit", "Cascade Method", "Electrical Characteristics")),
    assertions=(Assertion(code="ws2812_external_strip_output", message="J1 pin 2 is the only level-controlled DIN output; J1 pin 1 is a 5 V strip supply and pin 3 is its return. Supply current, cable EMI, and strip-side injection remain unmeasured."),),
    default=True,
)


# AL8860 typical buck loop: VIN -> 0.100 ohm sense resistor -> LED anode ->
# LED cathode -> inductor -> SW.  The sense resistor yields 1.0 A nominal.
# LED pad 3 is the emitter thermal pad/cathode land and shares that copper.
POWER_LED_AL8860 = _recipe(
    recipe="al8860-3w-power-led@1", family="3w-power-led", exact="AL8860-3W-POWER-LED",
    aliases=("3 W power LED",),
    ports=(Port(name="vin", direction="power"), Port(name="gnd", direction="power"), Port(name="enable", direction="input")),
    internal=("led_cathode", "switch", "set"),
    parts=(
        _part("driver", "U", "AL8860MP-13", "al8860:AL8860MP-13", "al8860:MSOP-8_L3.0-W3.0-P0.65-LS4.9-BL-EP1.8", mpn="AL8860MP-13"),
        _part("emitter", "D", "JNJ-LTJW0115W120/63MIL/6500-7000K", "jnj-ltjw0115w120-63mil-6500-7000k:JNJ-LTJW0115W120_63MIL_6500-7000K", "jnj-ltjw0115w120-63mil-6500-7000k:LED-SMD_3P-L3.5-W3.5-RD-C", mpn="JNJ-LTJW0115W120/63MIL/6500-7000K"),
        _part("inductor", "L", "15uH 3A", "inductor-15uh-3a:FXL0630-150-M", "inductor-15uh-3a:IND-SMD_L7.0-W6.6_FXL0630", mpn="FXL0630-150-M"),
        _part("catch_diode", "D", "SS14", "ss14:SS14", "ss14:SMA_L4.3-W2.6-LS5.0-RD", mpn="SS14"),
        _part("sense", "R", "0.1R", "Device:R", "Resistor_SMD:R_2512_6332Metric", mpn="RL2512FK-070R1L"),
        _passive("input_bulk", "C", "10uF"), _passive("decoupling", "C", "100nF"),
    ),
    pins=(
        Pin(role="driver", pin="1", net="set"), Pin(role="driver", pin="2", net="gnd"), Pin(role="driver", pin="3", net="gnd"), Pin(role="driver", pin="4", net="enable"), Pin(role="driver", pin="5", net="switch"), Pin(role="driver", pin="6", net="switch"), Pin(role="driver", pin="8", net="vin"), Pin(role="driver", pin="9", net="gnd"),
        Pin(role="emitter", pin="1", net="led_cathode"), Pin(role="emitter", pin="2", net="set"), Pin(role="emitter", pin="3", net="led_cathode"), Pin(role="inductor", pin="1", net="led_cathode"), Pin(role="inductor", pin="2", net="switch"), Pin(role="catch_diode", pin="1", net="vin"), Pin(role="catch_diode", pin="2", net="switch"), Pin(role="sense", pin="1", net="vin"), Pin(role="sense", pin="2", net="set"), Pin(role="input_bulk", pin="1", net="vin"), Pin(role="input_bulk", pin="2", net="gnd"), Pin(role="decoupling", pin="1", net="vin"), Pin(role="decoupling", pin="2", net="gnd"),
    ),
    no_connects=(NoConnect(role="driver", pin="7"),),
    source=Source(url="https://www.diodes.com/datasheet/download/AL8860.pdf", title="Diodes AL8860", revision="current", reviewed_date="2026-10-11", sections=("Typical Application Circuit", "Current Setting", "Thermal Considerations")),
    assertions=(Assertion(code="power_led_constant_current", message="The 0.100 ohm VIN-to-SET resistor establishes 1.0 A nominal; the emitter cathode and thermal pad share the inductor current path. Copper-area/thermal rise and emitted optical performance are unmeasured."),), safety=True, default=True,
)


_SEGMENTS = (("A", "14"), ("B", "16"), ("C", "20"), ("D", "23"), ("E", "21"), ("F", "15"), ("G", "17"), ("DP", "22"))
# CA/CC56 pin ordering from the Kingbright drawing: E,D,CC,C,DP,B,A,CC,F,G.
_DIGIT_PINS = (("E", "1"), ("D", "2"), ("C", "4"), ("DP", "5"), ("B", "6"), ("A", "7"), ("F", "9"), ("G", "10"))
MAX7219_FOUR_DIGIT = _recipe(
    recipe="max7219-four-digit-seven-segment@1", family="four-digit-seven-segment", exact="MAX7219-4DIGIT-CC",
    aliases=("4-digit 7-segment display",),
    ports=(Port(name="vdd_5v", direction="power"), Port(name="gnd", direction="power"), Port(name="din", direction="input"), Port(name="clk", direction="input"), Port(name="load", direction="input"), Port(name="dout", direction="output", required=False)),
    internal=tuple([f"seg_{name.lower()}" for name, _ in _SEGMENTS] + [f"digit_{n}" for n in range(4)] + ["iset", "serial_din", "serial_clk", "serial_load"]),
    parts=(
        _part("driver", "U", "MAX7219CWG", "Driver_LED:MAX7219", "Package_SO:SOIC-24W_7.5x15.4mm_P1.27mm", mpn="MAX7219CWG"),
        _part("level_shifter", "U", "SN74HCT245PWR-JSM", "sn74hct245:SN74HCT245PWR-JSM", "sn74hct245:TSSOP-20_L6.5-W4.4-P0.65-LS6.4-BL", mpn="SN74HCT245PWR-JSM"),
        _part("digit", "DS", "CC56-12EWA", "Display_Character:CC56-12EWA", "Display_7Segment:CA56-12EWA", quantity=4, mpn="CC56-12EWA"),
        _passive("iset", "R", "27k"), _passive("decoupling", "C", "100nF"), _passive("bulk", "C", "10uF"),
    ),
    pins=(
        Pin(role="driver", pin="1", net="serial_din"), Pin(role="driver", pin="2", net="digit_0"), Pin(role="driver", pin="4", net="gnd"), Pin(role="driver", pin="6", net="digit_2"), Pin(role="driver", pin="7", net="digit_3"), Pin(role="driver", pin="9", net="gnd"), Pin(role="driver", pin="11", net="digit_1"), Pin(role="driver", pin="12", net="serial_load"), Pin(role="driver", pin="13", net="serial_clk"), Pin(role="driver", pin="14", net="seg_a"), Pin(role="driver", pin="15", net="seg_f"), Pin(role="driver", pin="16", net="seg_b"), Pin(role="driver", pin="17", net="seg_g"), Pin(role="driver", pin="18", net="iset"), Pin(role="driver", pin="19", net="vdd_5v"), Pin(role="driver", pin="20", net="seg_c"), Pin(role="driver", pin="21", net="seg_e"), Pin(role="driver", pin="22", net="seg_dp"), Pin(role="driver", pin="23", net="seg_d"), Pin(role="driver", pin="24", net="dout"),
        Pin(role="level_shifter", pin="20", net="vdd_5v"), Pin(role="level_shifter", pin="10", net="gnd"), Pin(role="level_shifter", pin="1", net="vdd_5v"), Pin(role="level_shifter", pin="19", net="gnd"), Pin(role="level_shifter", pin="2", net="din"), Pin(role="level_shifter", pin="18", net="serial_din"), Pin(role="level_shifter", pin="3", net="clk"), Pin(role="level_shifter", pin="17", net="serial_clk"), Pin(role="level_shifter", pin="4", net="load"), Pin(role="level_shifter", pin="16", net="serial_load"),
        *(Pin(role="level_shifter", pin=str(pin), net="gnd") for pin in range(5, 10)),
        *(Pin(role="digit", index=i, pin=pin, net=f"seg_{segment.lower()}") for i in range(4) for segment, pin in _DIGIT_PINS), *(Pin(role="digit", index=i, pin="3", net=f"digit_{i}") for i in range(4)), *(Pin(role="digit", index=i, pin="8", net=f"digit_{i}") for i in range(4)),
        Pin(role="iset", pin="1", net="vdd_5v"), Pin(role="iset", pin="2", net="iset"), Pin(role="decoupling", pin="1", net="vdd_5v"), Pin(role="decoupling", pin="2", net="gnd"), Pin(role="bulk", pin="1", net="vdd_5v"), Pin(role="bulk", pin="2", net="gnd"),
    ),
    no_connects=(
        NoConnect(role="driver", pin="3"), NoConnect(role="driver", pin="5"),
        NoConnect(role="driver", pin="8"), NoConnect(role="driver", pin="10"),
        *(NoConnect(role="level_shifter", pin=str(pin)) for pin in range(11, 16)),
    ),
    source=Source(url="https://www.analog.com/media/en/technical-documentation/data-sheets/max7219-max7221.pdf", title="Analog Devices MAX7219/MAX7221", revision="19-3517; Rev 4", reviewed_date="2026-10-11", sections=("Pin Configuration", "Serial Interface", "Selecting RSET Resistor", "Typical Application Circuit")),
    assertions=(Assertion(code="four_distinct_digit_channels", message="A 74HCT245 translates three controller serial inputs to the 5 V MAX7219 domain; DIG0 through DIG3 each terminate at both common-cathode pads of a distinct CC56-12EWA digit. Segments are shared only through MAX7219 outputs."),), default=True,
)

# The existing two-HCT245 topology is repeated under the generator-facing matrix
# family so an unspecified LED matrix has one documented default rather than a
# fabricated panel geometry: 13 3.3-V logic inputs to a 16-pin HUB75 interface.
LED_MATRIX_HUB75 = _recipe(
    recipe="led-matrix-hub75-interface@1", family="led-matrix", exact="HUB75-LED-MATRIX",
    aliases=("LED matrix",),
    ports=(Port(name="vdd_5v", direction="power"), Port(name="gnd", direction="power"), *(Port(name=name, direction="input") for name in ("r0", "g0", "b0", "r1", "g1", "b1", "addr_a", "addr_b", "addr_c", "addr_d", "clk", "lat", "oe"))),
    internal=tuple(f"shifted{i}" for i in range(13)),
    parts=(
        _part("level_shifter", "U", "SN74HCT245PWR-JSM", "sn74hct245:SN74HCT245PWR-JSM", "sn74hct245:TSSOP-20_L6.5-W4.4-P0.65-LS6.4-BL", quantity=2, mpn="SN74HCT245PWR-JSM"),
        _part("connector", "J", "HUB75", "dc3-2p54-16pas:DC3-2.54-16PAS", "dc3-2p54-16pas:IDC-TH_16P-P2.54_C3406", mpn="DC3-2.54-16PAS"), _passive("decoupling", "C", "100nF", 2),
    ),
    pins=(
        *(Pin(role="level_shifter", index=u, pin="20", net="vdd_5v") for u in range(2)), *(Pin(role="level_shifter", index=u, pin="10", net="gnd") for u in range(2)), *(Pin(role="level_shifter", index=u, pin="1", net="vdd_5v") for u in range(2)), *(Pin(role="level_shifter", index=u, pin="19", net="gnd") for u in range(2)), *(Pin(role="decoupling", index=u, pin="1", net="vdd_5v") for u in range(2)), *(Pin(role="decoupling", index=u, pin="2", net="gnd") for u in range(2)),
        *(pin for i, (name, connector_pin) in enumerate((("r0", 1), ("g0", 2), ("b0", 3), ("r1", 5), ("g1", 6), ("b1", 7), ("addr_a", 9), ("addr_b", 10), ("addr_c", 11), ("addr_d", 12), ("clk", 13), ("lat", 14), ("oe", 15))) for pin in (Pin(role="level_shifter", index=i // 8, pin=str(2 + i % 8), net=name), Pin(role="level_shifter", index=i // 8, pin=str(18 - i % 8), net=f"shifted{i}"), Pin(role="connector", pin=str(connector_pin), net=f"shifted{i}"))), *(Pin(role="connector", pin=str(pin), net="gnd") for pin in (4, 8, 16)),
        *(Pin(role="level_shifter", index=1, pin=str(pin), net="gnd") for pin in (7, 8, 9)),
    ),
    no_connects=tuple(NoConnect(role="level_shifter", index=1, pin=str(pin)) for pin in (11, 12, 13)),
    source=Source(url="https://www.ti.com/lit/ds/symlink/sn74hct245.pdf", title="Texas Instruments SN74HCT245", revision="SCLS020H", reviewed_date="2026-10-11", sections=("Pin Configuration and Functions", "Device Functional Modes", "Layout Guidelines")),
    assertions=(Assertion(code="matrix_default_hub75_13_channels", message="Unspecified LED matrix defaults to a 1/32-scan HUB75 interface. It exports 13 separately shifted control channels, not an invented onboard pixel count; panel current, scan timing, EMC, and optical performance are unmeasured."),), default=True,
)

def expand_max7219_four_digit(resolved):
    """Do not leave a physical serial-output pin on an invented net."""
    from .registry import expand_static_definition

    definition = MAX7219_FOUR_DIGIT
    if "dout" not in resolved.selection.port_bindings:
        definition = definition.model_copy(
            update={
                "pins": tuple(
                    pin for pin in definition.pins
                    if not (pin.role == "driver" and pin.pin == "24")
                ),
                "no_connects": (
                    *definition.no_connects,
                    NoConnect(role="driver", pin="24"),
                ),
            }
        )
    return expand_static_definition(definition, resolved)


WAVE_D_DISPLAY_RECIPES = (WS2812_EXTERNAL_STRIP, POWER_LED_AL8860, MAX7219_FOUR_DIGIT, LED_MATRIX_HUB75)
