"""Reviewed device, physical-part, and family membership relations.

Identity equivalence is deliberately narrow.  Physical selections are exposed
separately so validators can require an exact, portable symbol/footprint pair
instead of treating a matching product-family label as construction evidence.
"""

from __future__ import annotations
import re

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping


@dataclass(frozen=True)
class ReviewedPart:
    """A reviewed physical candidate, or an explicit identity-only boundary.

    ``identity`` is an exact, normalized MPN/order code.  A portable candidate
    has non-``None`` ``bundle``, ``symbol``, and ``footprint`` fields; callers
    MUST use that symbol/footprint pair verbatim.  ``contacts`` enumerates
    signal contacts only (not mounting tabs).  An identity-only record models
    a reviewed package boundary but MUST NOT be selected for a BOM.
    """

    identity: str
    family: str
    package: str
    bundle: str | None
    symbol: str | None
    footprint: str | None
    physical_features: frozenset[str] = frozenset()
    function_keys: frozenset[str] = frozenset()
    contacts: tuple[str, ...] = ()
    manufacturer_sources: tuple[str, ...] = ()
    lcsc: str | None = None
    current_feedback: Mapping[str, object] = field(default_factory=dict)
    power_transfer: Mapping[str, object] = field(default_factory=dict)
    operating_limits: Mapping[str, object] = field(default_factory=dict)
    bootstrap: Mapping[str, object] = field(default_factory=dict)
    port_pins: Mapping[str, str] = field(default_factory=dict)
    support_network: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_portable_candidate(self) -> bool:
        """Whether this record names a complete vendored library pair."""
        return self.bundle is not None and self.symbol is not None and self.footprint is not None


# Manufacturer evidence reviewed 2026-09-16 is recorded per physical record.
# These metadata records are an inventory, not substitutions: the identity,
# package, and complete library pair all remain mandatory at selection time.
_NRF52840_QIAA_CONTACTS = tuple(
    "0 A8 A10 A12 A14 A16 A18 A20 A22 A23 "
    "B1 B3 B5 B7 B9 B11 B13 B15 B17 B19 B24 "
    "C1 D2 D23 E24 F2 F23 G1 H2 H23 J1 J24 K2 L1 L24 M2 N1 N24 "
    "P2 P23 R1 R24 T2 T23 U1 U24 V23 W1 W24 Y2 Y23 AA24 AB2 "
    "AC5 AC9 AC11 AC13 AC15 AC17 AC19 AC21 AC24 "
    "AD2 AD4 AD6 AD8 AD10 AD12 AD14 AD16 AD18 AD20 AD22 AD23".split()
)


REVIEWED_PARTS: tuple[ReviewedPart, ...] = (
    ReviewedPart(
        identity="kh-bnc50-3511",
        family="bnc-connector",
        package="right-angle through-hole BNC jack",
        bundle="bnc-pcb-jack",
        symbol="bnc-pcb-jack:KH-BNC50-3511",
        footprint="bnc-pcb-jack:ANT-TH_KH-BNC50-3511",
        physical_features=frozenset({"bnc-connector", "coaxial-connector"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C2837587.pdf",),
        lcsc="C2837587",
    ),
    # SHOU HAN DC005 — the 5.5/2.1 mm DC power jack the corpus asks for as a
    # "12 V DC barrel jack" (Surprise-me seeds 16/17/22).  The bundle is the
    # vendored easyeda2kicad pair the parts loader resolves for MPN ``DC005``,
    # so the record names that same symbol/footprint pair: physical_inventory_record
    # classifies a part only when (mpn, symbol, footprint) all match it exactly.
    # LCSC C431533 is the bundle's own catalog row (64 903 in stock at the installed
    # 2026-09-16 dump), and the datasheet is the citation for everything asserted
    # here: it publishes the DC 12 V / 0.5 A rating carried below and the -30..+70 C
    # range, and shows a 2.1 mm centre pin, a sleeve and a normally-closed switch
    # contact (drawing A1, 2023-02-12).  The classification comes from that part
    # identity and the catalog's "DC Power Connectors" category, never from a name
    # or footprint prefix rule.
    #
    # The vendored symbol exposes three contact numbers (1, 2, 3) and names none of
    # them; the vendor drawing numbers its *terminals* 1..4, so which number is the
    # centre, the sleeve and the switch is NOT asserted here.  Nothing in the
    # pipeline reads that mapping today (there is no barrel-jack lowerer: a draft
    # declares the jack's ports and the BOM selects this part), and a wrong guess
    # would be a fabricated electrical fact.  What matters for design is stated
    # without it: one of the three contacts is a normally-closed switch, so a design
    # that does not use it must leave it unconnected rather than tie it to the return.
    ReviewedPart(
        identity="dc005",
        family="barrel-jack",
        package=(
            "SHOU HAN DC005 right-angle through-hole DC power jack, 2.1 mm centre "
            "pin / 5.5 mm barrel, 3 contacts (centre, sleeve, normally-closed switch)"
        ),
        bundle="dc005-barrel-jack",
        symbol="dc005-barrel-jack:DC005_C431533",
        footprint="dc005-barrel-jack:DC-IN-TH_DC005",
        physical_features=frozenset(
            {
                "barrel-jack-connector",
                "power-connector",
                "wire-to-board-connector",
                # The family name is a feature on this record for the same reason the
                # screw-terminal and buck-converter records carry theirs: the BOM
                # work-unit derives a requirement's typed demand from its family
                # (`_required_physical_feature`), and a demand the library cannot name
                # silently stops pruning the unit's groups.
                "barrel-jack",
            }
        ),
        contacts=("1", "2", "3"),
        manufacturer_sources=(
            "https://datasheet.lcsc.com/datasheet/pdf/c1c86644cdb5448d228bda08350b2b1f.pdf",
        ),
        lcsc="C431533",
        # Contact functions from the datasheet's own SCHEMATIC (sheet 1 of the cited PDF):
        # contact 1 is the sprung tip/centre contact, contact 2 is the normally-closed switch
        # contact drawn closed to 1, contact 3 is the barrel/sleeve contact. The record holds
        # the *functions*; which net is the positive one stays the draft's statement.
        port_pins={"tip": "1", "switch": "2", "sleeve": "3"},
        operating_limits={"voltage_v": 12.0, "current_a": 0.5},
    ),
    ReviewedPart(
        identity="3296w-1-103lf",
        family="trim-potentiometer",
        package="3296W through-hole vertical trimmer",
        bundle="trim-pot-3296w-10k",
        symbol="trim-pot-3296w-10k:3296W-1-103LF",
        footprint="trim-pot-3296w-10k:RES-ADJ-TH_3296W",
        physical_features=frozenset({"trim-potentiometer", "adjustable-resistor"}),
        contacts=("1", "2", "3"),
        manufacturer_sources=("https://www.bourns.com/docs/product-datasheets/3296.pdf",),
        lcsc="C34846",
        # Pin functions from the cited datasheet's ordering block ("3296 W - 1 - 103 __ LF"),
        # which labels the three terminals with their travel ends: 1 = CCW end, 2 = wiper,
        # 3 = CW end.
        port_pins={"ccw": "1", "wiper": "2", "cw": "3"},
    ),
    ReviewedPart(
        identity="c0805c103j5gactu",
        family="timing-capacitor",
        package="0805 (2012 metric)",
        bundle="c0805c103j5gactu",
        symbol="c0805c103j5gactu:C0805C103J5GACTU",
        footprint="c0805c103j5gactu:C_0805_2012Metric",
        physical_features=frozenset({"timing-capacitor", "c0g-capacitor"}),
        function_keys=frozenset({"c0g-timing-capacitor-10nf"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://search.kemet.com/download/specsheet/C0805C103J5GACTU",),
        lcsc="C2167597",
        operating_limits={
            "capacitance_nf": 10.0,
            "tolerance_fraction": 0.05,
            "voltage_v": 50.0,
            "dielectric": "C0G/NP0",
        },
    ),
    ReviewedPart(
        identity="attiny402-ssn",
        family="attiny402",
        package="SOIC150, 8-pin",
        bundle="attiny402-ssn",
        symbol="attiny402-ssn:ATTINY402-SSN",
        footprint="attiny402-ssn:SOIC-8_L5.0-W4.0-P1.27-LS6.0-BL",
        physical_features=frozenset({"microcontroller", "updi-programmable", "attiny402"}),
        function_keys=frozenset({"ornament-controller"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=(
            "https://onlinedocs.microchip.com/oxy/GUID-5A56DB3A-31E1-4F46-984F-39186535C84E-en-US-7/GUID-CC06B137-262F-43C8-90FD-A14C59BF9C29.html",
            "https://jlcpcb.com/partdetail/MicrochipTech-ATTINY402SSN/C1339884",
        ),
        lcsc="C1339884",
        operating_limits={
            "supply_min_v": 1.8,
            "supply_max_v": 5.5,
            "temperature_min_c": -40,
            "temperature_max_c": 105,
            "maximum_cpu_mhz": 20,
            "carrier": "tube-or-tray",
        },
    ),
    ReviewedPart(
        identity="attiny402-ssnr",
        family="attiny402",
        package="SOIC150, 8-pin",
        bundle="attiny402-ssnr",
        symbol="attiny402-ssnr:ATTINY402-SSNR",
        footprint="attiny402-ssnr:SOIC-8_L5.0-W4.0-P1.27-LS6.0-BL",
        physical_features=frozenset({"microcontroller", "updi-programmable", "attiny402"}),
        function_keys=frozenset({"ornament-controller"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=(
            "https://onlinedocs.microchip.com/oxy/GUID-5A56DB3A-31E1-4F46-984F-39186535C84E-en-US-7/GUID-CC06B137-262F-43C8-90FD-A14C59BF9C29.html",
            "https://jlcpcb.com/partdetail/MicrochipTech-ATTINY402SSNR/C616056",
        ),
        lcsc="C616056",
        operating_limits={
            "supply_min_v": 1.8,
            "supply_max_v": 5.5,
            "temperature_min_c": -40,
            "temperature_max_c": 105,
            "maximum_cpu_mhz": 20,
            "carrier": "tape-and-reel",
            "same_device_package_grade_as": "attiny402-ssn",
        },
    ),
    ReviewedPart(
        identity="pj-320d",
        family="audio-jack-3-5mm",
        package="SMD 3.5 mm TRS jack",
        bundle="pj-320d",
        symbol="pj-320d:PJ-320D",
        footprint="pj-320d:AUDIO-SMD_PJ-320D-1",
        physical_features=frozenset({"audio-jack-3-5mm", "trs-audio-jack"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C431535.pdf",),
        lcsc="C431535",
    ),
    ReviewedPart(
        identity="sj1-3533ng",
        family="audio-jack-3-5mm",
        package="right-angle through-hole 3.5 mm TRS jack",
        bundle="sj1-3533ng",
        symbol="sj1-3533ng:SJ1-3533NG",
        footprint="sj1-3533ng:Jack_3.5mm_CUI_SJ1-3533NG_Horizontal",
        physical_features=frozenset({"audio-jack-3-5mm", "trs-audio-jack"}),
        function_keys=frozenset({"stereo-audio-input"}),
        contacts=("S", "T", "R"),
        manufacturer_sources=(
            "https://www.sameskydevices.com/product/interconnect/connectors/audio-connectors/jacks/sj1-3533ng",
            "https://www.sameskydevices.com/product/resource/sj1-353xng.pdf",
            "https://www.digikey.com/en/products/detail/same-sky-formerly-cui-devices/SJ1-3533NG/738701",
            "https://jlcpcb.com/partdetail/CUI-SJ13533NG/C4992459",
        ),
        lcsc="C4992459",
        operating_limits={
            "voltage_vdc": 12,
            "current_a": 1,
            "temperature_min_c": -25,
            "temperature_max_c": 85,
            "internal_switches": 0,
        },
        port_pins={
            "common": "S",
            "left": "T",
            "right": "R",
        },
    ),
    ReviewedPart(
        identity="kh-fg0.5-h2.0-24pin",
        family="fpc-connector",
        package="24-contact, 0.50 mm-pitch, bottom-contact flip-lock SMT FPC",
        bundle="fpc-24-0-5-kinghelm",
        symbol="fpc-24-0-5-kinghelm:KH-FG0.5-H2.0-24PIN",
        footprint="fpc-24-0-5-kinghelm:FPC-SMD_KH-FG0.5-H2.0-24PIN",
        physical_features=frozenset({"fpc-connector", "ffc-connector"}),
        contacts=tuple(str(number) for number in range(1, 25)),
        manufacturer_sources=(
            "https://www.kinghelm.net/fpc-connector-53730/55197.html",
            "https://www.kinghelm.net/applicationDetail/16048",
        ),
        lcsc="C2797213",
    ),
    ReviewedPart(
        identity="wj126v-5.0-02p-14-00a",
        family="screw-terminal",
        package="KANGNEX 1x02, 5.00 mm-pitch, through-hole screw terminal",
        bundle="screw-terminal-5mm-2p",
        symbol="screw-terminal-5mm-2p:WJ126V-5.0-2P",
        footprint="screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
        physical_features=frozenset({"screw-terminal", "terminal-block", "power-connector"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://www.lcsc.com/product-detail/Terminal-Blocks_WJ126V-5-0-2P_C8404.html",
            "https://www.lcsc.com/datasheet/C8404.pdf",
        ),
        lcsc="C8404",
        operating_limits={
            "pitch_mm": 5.0,
            "positions": 2,
            "voltage_v": 250,
            "current_a": 18,
            "temperature_min_c": -40,
            "temperature_max_c": 105,
        },
    ),
    ReviewedPart(
        # The demanded `jst-xh-connector` had no reviewed carrier, so a live brief that asked for
        # two XH connectors could not be built (owner's priority: a demanded part the library does
        # not carry must be added, not refused). The bundle was already vendored and marked
        # `production`; this is its reviewed record: real order code, real LCSC id, the datasheet,
        # its two signal contacts, and the class the demand names.
        identity="b2b-xh-a(lf)(sn)",
        family="jst-xh-connector",
        package="JST 1x02, 2.50 mm-pitch, top-entry shrouded XH header",
        bundle="b2b-xh-a-lf-sn",
        symbol="b2b-xh-a-lf-sn:B2B-XH-A",
        footprint="b2b-xh-a-lf-sn:CONN-TH_B2B-XH-A-LF-SN",
        physical_features=frozenset(
            {"jst-xh-connector", "wire-to-board-connector", "power-connector"}
        ),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://lcsc.com/product-detail/XH-Connectors_JST_B2B-XH-A-LF-SN_XHsocket-1-2P-"
            "pitch2-5mm_C158012.html",
        ),
        lcsc="C158012",
        operating_limits={
            "pitch_mm": 2.5,
            "positions": 2,
            "voltage_v": 250,
            "current_a": 3,
            "temperature_min_c": -25,
            "temperature_max_c": 85,
        },
    ),
    ReviewedPart(
        identity="wj126v-5.0-03p-14-00a",
        family="screw-terminal",
        package="KANGNEX 1x03, 5.00 mm-pitch, through-hole screw terminal",
        bundle="screw-terminal-5mm-3p",
        symbol="screw-terminal-5mm-3p:WJ126V-5.0-3P",
        footprint="screw-terminal-5mm-3p:CONN-TH_3P-P5.00_WJ126V-5.0-3P",
        physical_features=frozenset({"screw-terminal", "terminal-block", "power-connector"}),
        contacts=("1", "2", "3"),
        manufacturer_sources=(
            "https://www.lcsc.com/product-detail/Terminal-Blocks_WJ126V-5-0-3P_C8401.html",
            "https://www.lcsc.com/datasheet/C8401.pdf",
        ),
        lcsc="C8401",
        operating_limits={
            "pitch_mm": 5.0,
            "positions": 3,
            "voltage_v": 250,
            "current_a": 18,
            "temperature_min_c": -40,
            "temperature_max_c": 105,
        },
    ),
    ReviewedPart(
        identity="dshp03tsget",
        family="three-position-dip-switch",
        package="YE 3-channel 1.27 mm-pitch SMD DIP switch",
        bundle="dip-switch-3pos",
        symbol="dip-switch-3pos:DSHP03TSGER",
        footprint="dip-switch-3pos:SW-SMD_6P-L5.4-W5.4-P1.27-LS8.4",
        physical_features=frozenset({"dip-switch", "three-channel-selector"}),
        contacts=("1", "2", "3", "4", "5", "6"),
        manufacturer_sources=(
            "https://www.lcsc.com/product-detail/DIP-Switches_1-27mm-3_C50844.html",
        ),
        lcsc="C50844",
    ),
    ReviewedPart(
        identity="ss13d07vg4",
        family="three-position-selector",
        package="SHOUHAN 4-contact, 3-position through-hole slide switch with two frame tabs",
        bundle="ss13d07vg4",
        symbol="ss13d07vg4:SS13D07VG4",
        footprint="ss13d07vg4:SW-TH_SS13D07VG4",
        physical_features=frozenset({"three-position-selector", "sp3t-selector"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=(
            "https://datasheet.lcsc.com/datasheet/pdf/39fcef34462917ff9922c33e708581d0.pdf?productCode=C2681578",
        ),
        lcsc="C2681578",
        operating_limits={"voltage_vdc": 30, "current_a": 0.5},
        port_pins={
            "throw_1": "1",
            "common": "2",
            "throw_2": "3",
            "throw_3": "4",
        },
        support_network={
            "conductive_frame_pads": ("5", "6"),
            "frame_pads_must_be_grounded": True,
        },
    ),
    ReviewedPart(
        identity="bme280",
        family="environmental-sensor",
        package="Bosch 8-pad LGA, 2.5 x 2.5 mm",
        bundle="bme280-bosch-sensor",
        symbol="bme280-bosch-sensor:BME280",
        footprint="bme280-bosch-sensor:LGA-8_BME280_BL",
        physical_features=frozenset({"environmental-sensor", "i2c-sensor"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=(
            "https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme280-ds002.pdf",
        ),
        lcsc="C92489",
        port_pins={
            "ground": "1",
            "chip_select": "2",
            "sda": "3",
            "scl": "4",
            "sdo": "5",
            "vddio": "6",
            "vdd": "8",
        },
    ),
    ReviewedPart(
        identity="sm04b-srss-tb(lf)(sn)",
        family="qwiic-i2c-connector",
        package="JST SH 1x04, 1.00 mm-pitch side-entry SMD receptacle",
        bundle="jst-sh-4p-qwiic",
        symbol="jst-sh-4p-qwiic:SM04B-SRSS-TB",
        footprint="jst-sh-4p-qwiic:CONN-SMD_4P-P1.00_SM04B-SRSS-TB-LF-SN",
        physical_features=frozenset({"qwiic-connector", "i2c-connector"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=(
            "https://www.lcsc.com/product-detail/Others_JST-Sales-America__JST-Sales-America-SM04B-SRSS-TB-LF-SN_C160404.html",
        ),
        lcsc="C160404",
        port_pins={"ground": "1", "vcc": "2", "sda": "3", "scl": "4"},
    ),
    ReviewedPart(
        identity="mcp6004-i/sl",
        family="quad-operational-amplifier",
        package="Microchip SOIC-14",
        bundle="mcp6004",
        symbol="mcp6004:MCP6004-I_SL",
        footprint="mcp6004:SOIC-14_L8.7-W3.9-P1.27-LS6.0-BL",
        physical_features=frozenset({"quad-operational-amplifier"}),
        contacts=tuple(str(number) for number in range(1, 15)),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C1346056.pdf",),
        lcsc="C1346056",
        port_pins={"vdd": "4", "vss": "11"},
    ),
    ReviewedPart(
        identity="pc817c-s",
        family="optocoupler",
        package="UMW 4-pin SOP optocoupler",
        bundle="pc817c-s",
        symbol="pc817c-s:PC817C-S",
        footprint="pc817c-s:SOP-4_L6.5-W4.6-P2.54-LS10.3-TL",
        physical_features=frozenset({"optocoupler", "isolated-interface"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C3008369.pdf",),
        lcsc="C3008369",
        port_pins={"anode": "1", "cathode": "2", "emitter": "3", "collector": "4"},
    ),
    ReviewedPart(
        identity="adum1301arwz-rl",
        family="digital-isolator",
        package="Analog Devices wide-body SOIC-16",
        bundle="adum1301arwz-rl",
        symbol="adum1301arwz-rl:ADUM1301ARWZ-RL",
        footprint="adum1301arwz-rl:SOIC-16W_L10.3-W7.5-P1.27-LS10.3-BL",
        physical_features=frozenset({"digital-isolator", "isolated-interface"}),
        contacts=tuple(str(number) for number in range(1, 17)),
        manufacturer_sources=(
            "https://www.analog.com/media/en/technical-documentation/data-sheets/ADuM1300_1301.pdf",
        ),
        lcsc="C22261",
        port_pins={
            "vdd1": "1",
            "gnd1": "2",
            "ve1": "7",
            "gnd1_return": "8",
            "gnd2": "9",
            "ve2": "10",
            "gnd2_return": "15",
            "vdd2": "16",
        },
    ),
    ReviewedPart(
        identity="b0509s-1wr3",
        family="isolated-dc-dc-converter",
        package="EVISUN SIP-4, 11.68 x 10.20 mm",
        bundle="b0509s-1wr3",
        symbol="b0509s-1wr3:B0509S-1WR3",
        footprint="b0509s-1wr3:SIP-4_L11.68-W10.20-P2.54-BL",
        physical_features=frozenset({"isolated-dc-dc-converter", "isolated-interface"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C7500906.pdf",),
        lcsc="C7500906",
        # The reviewed 1 W isolated module's input window: the B_S-1WR3 series is a
        # 5 V nominal part, and this order code is published as a 4.5-5.5 V input
        # (LCSC C7500906: "Isolated Module DC DC Converter 9 V 112 mA 4.5 V~5.5 V
        # Input").  Without the window a design containing it cannot prove its
        # input rail is inside the reviewed range (§9.38).
        operating_limits={
            "input_v": 5.0,
            "input_voltage_min_v": 4.5,
            "input_voltage_max_v": 5.5,
            "output_v": 9.0,
            "output_power_w": 1.0,
            "isolation_vdc": 1500,
        },
        port_pins={
            "input_negative": "1",
            "input_positive": "2",
            "output_negative": "3",
            "output_positive": "4",
        },
        # The module IS the complete converter: the LCSC row (C7500906) is a 1 W
        # isolated module, 4.5-5.5 V in to 9 V / 112 mA out at 1.5 kV isolation, so
        # its datasheet path is the module-level one (input pin 2 -> output pin 4)
        # and its return (pin 1 -> pin 3) is the isolated reference domain §9.39
        # requires a record to declare.  Both pins are named by number because the
        # review is of the module's own pin table, not of a pin-name convention.
        power_transfer={
            "isolated": True,
            "paths": [
                {"from_pin": "2", "to_pin": "4", "return_from_pin": "1", "return_to_pin": "3"}
            ],
        },
    ),
    ReviewedPart(
        identity="hs96l03w2c03",
        family="i2c-oled-display",
        package="0.96 inch 128x64 SSD1315 OLED module, 4-pin through-hole header",
        bundle="hs96l03w2c03",
        symbol="hs96l03w2c03:HS96L03W2C03",
        footprint="hs96l03w2c03:OLED-TH_L27.8-W27.2-P2.54_C9900033791",
        physical_features=frozenset({"i2c-oled-display", "oled-display"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C5248080.pdf",),
        lcsc="C5248080",
        port_pins={"ground": "1", "vcc": "2", "scl": "3", "sda": "4"},
    ),
    ReviewedPart(
        identity="stm32f103c8t6",
        family="stm32f103",
        package="STMicroelectronics LQFP-48, 7 x 7 mm, 0.50 mm pitch",
        bundle="stm32f103c8t6",
        symbol="stm32f103c8t6:STM32F103C8T6",
        footprint="stm32f103c8t6:LQFP-48_L7.0-W7.0-P0.50-LS9.0-BL",
        physical_features=frozenset({"microcontroller", "stm32-mcu"}),
        contacts=tuple(str(number) for number in range(1, 49)),
        manufacturer_sources=("https://www.st.com/resource/en/datasheet/stm32f103c8.pdf",),
        lcsc="C8734",
    ),
    ReviewedPart(
        identity="e6c0805wway1uda(1.1t m)",
        family="warm-white-led",
        package="EKINGLUX 0805 (2012 metric) SMD LED",
        bundle="e6c0805wway1uda-1-1t-m",
        symbol="e6c0805wway1uda-1-1t-m:E6C0805WWAY1UDA",
        footprint="e6c0805wway1uda-1-1t-m:LED0805-RD_WHITE",
        physical_features=frozenset({"warm-white-led", "led-0805", "indicator-led"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://www.lcsc.com/product-detail/Light-Emitting-Diodes-LED_EKINGLUX-E6C0805WWAY1UDA-1-1T-M_C6916225.html",
            "https://www.lcsc.com/datasheet/C6916225.pdf",
        ),
        lcsc="C6916225",
        operating_limits={
            "color_temperature_min_k": 2800,
            "color_temperature_max_k": 3200,
            "forward_voltage_v": 3.0,
            "forward_current_ma": 20,
        },
        port_pins={"cathode": "1", "anode": "2"},
    ),
    ReviewedPart(
        identity="nrf52840-qiaa-r",
        family="nrf52840",
        package="Nordic 73-contact AQFN, 7.0 x 7.0 mm, 0.50 mm pitch, 4.85 mm exposed pad",
        bundle="nrf52840-qiaa-r",
        symbol="nrf52840-qiaa-r:NRF52840-QIAA-R",
        footprint="nrf52840-qiaa-r:AQFN-73_L7.0-W7.0-P0.50-BL-EP4.8",
        physical_features=frozenset({"microcontroller", "bluetooth-le-soc", "nrf52840"}),
        contacts=_NRF52840_QIAA_CONTACTS,
        manufacturer_sources=(
            "https://lcsc.com/product-detail/Others_Nordic-Semicon_NRF52840-QIAA-R_Nordic-Semicon-NRF52840-QIAA-R_C190794.html",
        ),
        lcsc="C190794",
        port_pins={
            "antenna": "H23",
            "reset": "AC13",
            "swdclk": "AA24",
            "swdio": "AC24",
            "usb_minus": "AD4",
            "usb_plus": "AD6",
            "vdd": "A22",
            "ground": "B7",
        },
    ),
    ReviewedPart(
        identity="nrf52840-qiaa-r7",
        family="nrf52840",
        package="Nordic 73-contact AQFN, 7.0 x 7.0 mm, 0.50 mm pitch, 4.85 mm exposed pad; R7 is the reel order code for this package",
        bundle="nrf52840-qiaa-r",
        symbol="nrf52840-qiaa-r:NRF52840-QIAA-R",
        footprint="nrf52840-qiaa-r:AQFN-73_L7.0-W7.0-P0.50-BL-EP4.8",
        physical_features=frozenset({"microcontroller", "bluetooth-le-soc", "nrf52840"}),
        contacts=_NRF52840_QIAA_CONTACTS,
        manufacturer_sources=(
            "https://lcsc.com/product-detail/Others_Nordic-Semicon_NRF52840-QIAA-R_Nordic-Semicon-NRF52840-QIAA-R_C190794.html",
        ),
        lcsc="C190794",
        port_pins={
            "antenna": "H23",
            "reset": "AC13",
            "swdclk": "AA24",
            "swdio": "AC24",
            "usb_minus": "AD4",
            "usb_plus": "AD6",
            "vdd": "A22",
            "ground": "B7",
        },
    ),
    ReviewedPart(
        identity="h2u38d1e1b0100",
        family="chip-antenna-2g4",
        package="Unictron CW324S ceramic chip antenna, 3.2 x 1.6 x 1.3 mm",
        bundle="unictron-h2u38d1e1b0100",
        symbol="unictron-h2u38d1e1b0100:H2U38D1E1B0100",
        footprint="unictron-h2u38d1e1b0100:ANT-SMD_L3.2-W1.6-3",
        physical_features=frozenset({"chip-antenna", "2.4ghz-antenna", "bluetooth-antenna"}),
        contacts=("1", "2"),
        port_pins={"feed": "1", "nc": "2"},
        manufacturer_sources=(
            "https://www.unictron.com/wp-content/uploads/datasheet/H2U38D1E1B0100.pdf",
            "https://www.unictron.com/wireless-communications/product/h2u38d1e1b0100/",
        ),
        lcsc="C6569546",
        operating_limits={
            "frequency_low_mhz": 2400,
            "frequency_high_mhz": 2500,
            "impedance_ohm": 50,
            "maximum_input_power_w": 2,
            "operating_temperature_min_c": -40,
            "operating_temperature_max_c": 85,
            "height_mm": 1.3,
        },
    ),
    ReviewedPart(
        identity="aonr21357",
        family="p-channel-highside-mosfet",
        package="AOS 3.0 x 3.0 mm DFN-8 with exposed drain thermal pad",
        bundle="aonr21357",
        symbol="aonr21357:AONR21357",
        footprint="aonr21357:DFN-8_L3.0-W3.0-P0.65-BL",
        physical_features=frozenset({"p-channel-mosfet", "highside-switch", "thermal-pad"}),
        contacts=tuple(str(number) for number in range(1, 10)),
        manufacturer_sources=(
            "https://www.lcsc.com/product-detail/MOSFETs_Alpha-Omega-Semicon-AONR21357_C431196.html",
            "https://www.lcsc.com/datasheet/C431196.pdf",
        ),
        lcsc="C431196",
        operating_limits={
            "vds_max_v": 30.0,
            "continuous_drain_a": 34.0,
            "rds_on_ohm_at_vgs_minus_4p5v": 0.0123,
            "vgs_abs_max_v": 25.0,
            "power_dissipation_w": 30.0,
        },
        power_transfer={"from_pin": "1", "to_pin": "5"},
        port_pins={"source": "1", "gate": "4", "drain": "5", "thermal_drain": "9"},
        support_network={
            "source_pins": ("1", "2", "3"),
            "drain_pins": ("5", "6", "7", "8", "9"),
            "thermal_pad": "9",
        },
    ),
    ReviewedPart(
        identity="wra2412s-3wr2",
        family="dual-output-dc-dc-converter",
        package="SIP-8, 22 x 9.5 x 12 mm through-hole module",
        bundle="wra2412s-3wr2",
        symbol="wra2412s-3wr2:WRA2412S-3WR2",
        footprint="wra2412s-3wr2:Converter_DCDC_ReSine_WRAxxxxS-3WR2_THT",
        physical_features=frozenset({"dual-output-dc-dc-converter", "isolated-dc-dc-converter"}),
        function_keys=frozenset({"24v-to-plus-minus-12v"}),
        contacts=("1", "2", "3", "5", "6", "7", "8"),
        manufacturer_sources=(
            "https://www.lcsc.com/datasheet/C20617261.pdf",
            "https://jlcpcb.com/partdetail/ReSinePS-WRA2412S3WR2/C20617261",
            "https://www.lcsc.com/product-detail/Isolated-Power-Modules_ReSine-PS-WRA2412S-3WR2_C20617261.html",
        ),
        lcsc="C20617261",
        operating_limits={
            "input_min_v": 18.0,
            "input_max_v": 36.0,
            "nominal_input_v": 24.0,
            "positive_output_v": 12.0,
            "negative_output_v": -12.0,
            "output_current_per_rail_min_a": 0.0125,
            "output_current_per_rail_max_a": 0.125,
            "output_power_w": 3.0,
            "isolation_vdc": 1500,
            "output_capacitance_max_uf": 470,
            "temperature_min_c": -40,
            "temperature_max_c": 105,
        },
        power_transfer={
            "isolated": True,
            "paths": [
                {"from_pin": "VIN", "to_pin": "+VO"},
                {"from_pin": "VIN", "to_pin": "-VO"},
            ],
        },
        port_pins={
            "input_return": "GND",
            "input_positive": "VIN",
            "control": "CTRL",
            "no_connect": "NC",
            "output_positive": "+VO",
            "output_common": "0V",
            "output_negative": "-VO",
        },
    ),
    ReviewedPart(
        identity="keystone-8734",
        family="binding-post",
        package="90-degree through-hole, M2.5 screw-clamp binding post",
        bundle="keystone-8734-binding-post",
        symbol="keystone-8734-binding-post:Keystone_8734",
        footprint="keystone-8734-binding-post:Keystone_8734",
        physical_features=frozenset({"binding-post", "screw-clamp-terminal"}),
        function_keys=frozenset({"bare-wire-binding-post"}),
        contacts=("1",),
        manufacturer_sources=("https://www.keyelco.com/product.cfm/product_id/14242",),
        operating_limits={"wire_awg_min": 14, "wire_awg_max": 22},
    ),
    ReviewedPart(
        identity="dayton-lw18-50",
        family="air-core-inductor",
        package="43 mm OD, 16 mm high radial through-hole coil",
        bundle="dayton-lw18-50",
        symbol="dayton-lw18-50:Dayton_LW18-50",
        footprint="dayton-lw18-50:Dayton_LW18-50",
        physical_features=frozenset({"air-core-inductor", "speaker-crossover-inductor"}),
        function_keys=frozenset({"speaker-crossover-inductor-0.5mh"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://www.daytonaudio.com/product/1327/lw18-50-0-50mh-18-awg-perfect-layer-inductor",
        ),
        operating_limits={
            "inductance_mh": 0.5,
            "tolerance_fraction": 0.03,
            "dc_resistance_ohm": 0.33,
            "rms_power_w": 300,
        },
    ),
    ReviewedPart(
        identity="mkp20685j2g362230",
        family="speaker-crossover-capacitor",
        package="radial metallized polypropylene, 48.0 mm pitch",
        bundle="kyet-mkp20685j2g362230",
        symbol="kyet-mkp20685j2g362230:KYET_MKP20685J2G362230",
        footprint="kyet-mkp20685j2g362230:KYET_MKP20685J2G362230_P48.0mm",
        physical_features=frozenset({"film-capacitor", "speaker-crossover-capacitor"}),
        function_keys=frozenset({"speaker-crossover-capacitor-6.8uf"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://datasheet.lcsc.com/datasheet/pdf/988445a2fd1023836a8a59546d7ba7fe.pdf?productCode=C521054",
        ),
        lcsc="C521054",
        operating_limits={
            "capacitance_uf": 6.8,
            "tolerance_fraction": 0.05,
            "dc_voltage_v": 400,
            "dielectric": "metallized-polypropylene",
        },
    ),
    ReviewedPart(
        identity="mkp1848510924k2",
        family="speaker-crossover-capacitor",
        package="radial metallized polypropylene, 27.5 mm pitch",
        bundle="vishay-mkp1848510924k2",
        symbol="vishay-mkp1848510924k2:VISHAY_MKP1848510924K2",
        footprint="vishay-mkp1848510924k2:VISHAY_MKP1848510924K2_P27.5mm",
        physical_features=frozenset({"film-capacitor", "speaker-crossover-capacitor"}),
        function_keys=frozenset({"speaker-crossover-capacitor-1uf"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://datasheet.lcsc.com/datasheet/pdf/e1923bec7fca44547afd891a5eadb594.pdf?productCode=C3802212",
        ),
        lcsc="C3802212",
        operating_limits={
            "capacitance_uf": 1.0,
            "tolerance_fraction": 0.05,
            "dc_voltage_v": 1200,
            "dielectric": "metallized-polypropylene",
        },
    ),
    ReviewedPart(
        identity="tps2553dbvr",
        family="current-limited-power-switch",
        package="DBV SOT-23-6",
        bundle="tps2553dbvr",
        symbol="tps2553dbvr:TPS2553DBVR",
        footprint="tps2553dbvr:SOT-23-6_DBV",
        physical_features=frozenset({"current-limited-power-switch", "usb-current-limiter"}),
        function_keys=frozenset({"usb-independent-current-limit"}),
        contacts=("1", "2", "3", "4", "5", "6"),
        manufacturer_sources=("https://www.ti.com/lit/ds/symlink/tps2553.pdf",),
        lcsc="C55266",
        operating_limits={"vin_min_v": 2.5, "vin_max_v": 6.5, "continuous_current_a": 1.5},
        port_pins={
            "input": "IN",
            "ground": "GND",
            "enable": "EN",
            "fault": "FAULT",
            "limit": "ILIM",
            "output": "OUT",
        },
        support_network={
            "ilim_resistor_ohm_min": 15000,
            "ilim_resistor_ohm_max": 232000,
            "input_decoupling_uf_min": 0.1,
        },
    ),
    ReviewedPart(
        identity="al8860mp-13",
        family="constant-current-led-driver",
        package="MSOP-8EP",
        bundle="al8860",
        symbol="al8860:AL8860MP-13",
        footprint="al8860:MSOP-8_L3.0-W3.0-P0.65-LS4.9-BL-EP1.8",
        physical_features=frozenset({"constant-current-led-driver"}),
        function_keys=frozenset({"usb-led-current-regulator-1a"}),
        contacts=("1", "2", "3", "4", "5", "6", "7", "8", "9"),
        manufacturer_sources=("https://www.diodes.com/datasheet/download/AL8860.pdf",),
        lcsc="C500782",
        current_feedback={
            "sense_pin": "SET",
            "reference_pin": "VIN",
            "sense_voltage_v": 0.1,
            "sense_tolerance": 0.04,
            "topology": "high_side_sense_low_side_switch",
        },
        power_transfer={"from_pin": "SW", "to_pin": "GND"},
        port_pins={
            "input": "VIN",
            "ground": "GND",
            "switch": "SW",
            "set": "SET",
            "control": "CTRL",
        },
        operating_limits={
            "vin_min_v": 4.5,
            "vin_max_v": 40.0,
            "continuous_output_a": 1.5,
        },
        support_network={
            "input_decoupling": {
                "positive_pin": "VIN",
                "negative_pin": "GND",
                "capacitance_min_uf": 10.0,
                "dielectric": "X7R",
            },
            "catch_diode": {
                "cathode_pin": "VIN",
                "anode_pin": "SW",
            },
        },
    ),
    ReviewedPart(
        identity="tps54331ddar",
        family="buck-converter",
        package="SO PowerPAD, 8-pin DDA",
        bundle="tps54331",
        symbol="tps54331:TPS54331DDAR",
        footprint="tps54331:SOIC-8_L4.9-W3.9-P1.27-LS6.0-BL-EP",
        physical_features=frozenset({"buck-converter"}),
        function_keys=frozenset({"adjustable-buck"}),
        contacts=("1", "2", "3", "4", "5", "6", "7", "8", "9"),
        manufacturer_sources=("https://www.ti.com/lit/ds/symlink/tps54331.pdf",),
        lcsc="C90761",
        operating_limits={
            "vin_min_v": 3.5,
            "vin_max_v": 28.0,
            "continuous_output_a": 3.0,
        },
        bootstrap={
            "positive_pin": "BOOT",
            "negative_pin": "PH",
            "capacitance_uf": 0.1,
        },
        power_transfer={"from_pin": "VIN", "to_pin": "PH"},
        port_pins={
            "input": "VIN",
            "ground": "GND",
            "switch": "PH",
            "feedback": "VSENSE",
        },
    ),
    ReviewedPart(
        identity="tps5430",
        family="buck-converter",
        package="HSOIC PowerPAD, 8-pin DDA",
        bundle=None,
        symbol=None,
        footprint=None,
        manufacturer_sources=("https://www.ti.com/lit/ds/symlink/tps5430.pdf",),
        operating_limits={
            "vin_min_v": 5.5,
            "vin_max_v": 36.0,
            "continuous_output_a": 3.0,
        },
        bootstrap={
            "positive_pin": "BOOT",
            "negative_pin": "PH",
            "capacitance_uf": 0.01,
        },
        power_transfer={"from_pin": "VIN", "to_pin": "PH"},
        port_pins={
            "input": "VIN",
            "ground": "GND",
            "switch": "PH",
            "feedback": "VSENSE",
        },
    ),
    ReviewedPart(
        identity="tps5430dda",
        family="buck-converter",
        package="HSOIC PowerPAD, 8-pin DDA",
        bundle=None,
        symbol=None,
        footprint=None,
        manufacturer_sources=("https://www.ti.com/lit/ds/symlink/tps5430.pdf",),
        operating_limits={
            "vin_min_v": 5.5,
            "vin_max_v": 36.0,
            "continuous_output_a": 3.0,
        },
        bootstrap={
            "positive_pin": "BOOT",
            "negative_pin": "PH",
            "capacitance_uf": 0.01,
        },
        power_transfer={"from_pin": "VIN", "to_pin": "PH"},
        port_pins={
            "input": "VIN",
            "ground": "GND",
            "switch": "PH",
            "feedback": "VSENSE",
        },
    ),
    # The buck-3a row emits the stock KiCad pair Regulator_Switching:TPS5430DDA +
    # its HSOP-8 land pattern for this TI order code (TI SLVS632L; the row's
    # part_inventory fact and the reviewed sourceability note both name it, with
    # LCSC C9864 in the read-only JLC/LCSC dump).  The order code therefore
    # carries that exact pair instead of staying identity-only: the "tps5430"
    # device designation and the "tps5430dda" package designation above remain
    # the identity-only boundaries, and nothing here is package-inferred.
    ReviewedPart(
        identity="tps5430ddar",
        family="buck-converter",
        package="HSOIC PowerPAD, 8-pin DDA; tape-and-reel carrier",
        bundle="kicad-standard",
        symbol="Regulator_Switching:TPS5430DDA",
        footprint="Package_SO:HSOP-8-1EP_3.9x4.9mm_P1.27mm_EP2.41x3.1mm",
        physical_features=frozenset({"buck-converter"}),
        contacts=("1", "2", "3", "4", "5", "6", "7", "8", "9"),
        manufacturer_sources=("https://www.ti.com/lit/ds/symlink/tps5430.pdf",),
        lcsc="C9864",
        operating_limits={
            "vin_min_v": 5.5,
            "vin_max_v": 36.0,
            "continuous_output_a": 3.0,
        },
        bootstrap={
            "positive_pin": "BOOT",
            "negative_pin": "PH",
            "capacitance_uf": 0.01,
        },
        power_transfer={"from_pin": "VIN", "to_pin": "PH"},
        port_pins={
            "input": "VIN",
            "ground": "GND",
            "switch": "PH",
            "feedback": "VSENSE",
        },
    ),
    ReviewedPart(
        identity="stm32l072czu6",
        family="stm32l0",
        package="UFQFPN-48, 7 x 7 mm; not LQFP-48",
        bundle=None,
        symbol=None,
        footprint=None,
        manufacturer_sources=("https://www.st.com/resource/en/datasheet/stm32l072cz.pdf",),
    ),
    # --- LoRa node reference hardware (reviewed 2026-09-16) -------------------
    # The original lora-node brief names an unqualified "SX1276 module", an
    # "STM32L0", an "SMA antenna connector" and a "screw-terminal sensor input".
    # These records make each one exact hardware instead of prose. Semtech's own
    # SX1276RF1JAS/RF1IAS carrier modules are legacy and not recommended for new
    # designs, so the reviewed module is a currently-manufactured SX1276 design.
    ReviewedPart(
        identity="dl-rfm95-868m",
        family="sx1276-module",
        package="DreamLNK 16 x 16 x 1.8 mm SMD-16 LoRa module built on the Semtech SX1276",
        bundle="dl-rfm95-868m",
        symbol="dl-rfm95-868m:DL-RFM95-868M",
        footprint="dl-rfm95-868m:COMM-SMD_RFM95W",
        physical_features=frozenset({"sx1276-module", "lora-module", "spi-radio-module"}),
        contacts=tuple(str(number) for number in range(1, 17)),
        manufacturer_sources=(
            "https://www.dreamlnk.com/en/DL-RFM95.html",
            "https://www.lcsc.com/datasheet/C2844472.pdf",
        ),
        lcsc="C2844472",
        operating_limits={
            "supply_min_v": 1.8,
            "supply_max_v": 3.7,
            "supply_typical_v": 3.3,
            "frequency_min_mhz": 820,
            "frequency_max_mhz": 1020,
            "band_mhz": 868,
            "tx_power_max_dbm": 19.5,
        },
        port_pins={
            "ground": "1",
            "miso": "2",
            "mosi": "3",
            "sck": "4",
            "nss": "5",
            "reset": "6",
            "antenna": "9",
            "vdd": "13",
            "dio0": "14",
        },
    ),
    ReviewedPart(
        identity="stm32l031k6t6",
        family="stm32l0",
        package="STMicroelectronics LQFP-32, 7 x 7 mm, 0.80 mm pitch",
        bundle="kicad-standard",
        symbol="MCU_ST_STM32L0:STM32L031K6Tx",
        footprint="Package_QFP:LQFP-32_7x7mm_P0.8mm",
        physical_features=frozenset({"microcontroller", "stm32-mcu"}),
        contacts=tuple(str(number) for number in range(1, 33)),
        manufacturer_sources=(
            "https://www.st.com/resource/en/datasheet/stm32l031k4.pdf",
            "https://www.st.com/en/microcontrollers-microprocessors/stm32l031k6.html",
        ),
        lcsc="C94085",
        operating_limits={
            "supply_min_v": 1.65,
            "supply_max_v": 3.6,
            "max_clock_mhz": 32,
        },
        port_pins={
            "vdd": "1",
            "nrst": "4",
            "vdda": "5",
            "vss": "16",
            "swdio": "23",
            "swclk": "24",
            "boot0": "31",
        },
    ),
    ReviewedPart(
        identity="132289",
        family="sma-connector",
        package="Amphenol RF 132289 50-ohm SMA end-launch jack for a 1.57 mm board edge",
        bundle="kicad-standard",
        symbol="Connector:Conn_Coaxial",
        footprint="Connector_Coaxial:SMA_Amphenol_132289_EdgeMount",
        physical_features=frozenset({"sma-connector", "coaxial-connector", "rf-connector"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://www.amphenolrf.com/library/download/link/link_id/592024/",
            "https://www.lcsc.com/datasheet/C3172723.pdf",
        ),
        lcsc="C3172723",
        operating_limits={"impedance_ohm": 50, "frequency_max_ghz": 18},
        port_pins={"center": "1", "shell": "2"},
    ),
    ReviewedPart(
        identity="wj126v-5.0-04p-14-00a",
        family="screw-terminal",
        package="KANGNEX 1x04, 5.00 mm-pitch, through-hole screw terminal",
        bundle="screw-terminal-5mm-4p",
        symbol="screw-terminal-5mm-4p:WJ126V-5.0-4P",
        footprint="screw-terminal-5mm-4p:CONN-TH_4P-P5.00_WJ126V-5.0-4P-1",
        physical_features=frozenset({"screw-terminal", "terminal-block", "power-connector"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=(
            "https://www.china-wj.com/cms/en/prod_pdf/id/18.html",
            "https://www.lcsc.com/datasheet/C2931152.pdf",
        ),
        lcsc="C2931152",
        operating_limits={"pitch_mm": 5.0, "positions": 4},
    ),
    ReviewedPart(
        identity="grm188r71c104ka01d",
        family="capacitor-0603",
        package="Murata 0603 (1608 metric) X7R MLCC",
        bundle="kicad-standard",
        symbol="Device:C",
        footprint="Capacitor_SMD:C_0603_1608Metric",
        physical_features=frozenset({"capacitor-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://www.murata.com/en-global/products/productdetail?partno=GRM188R71C104KA01D",
        ),
        lcsc="C45000",
        operating_limits={"capacitance_nf": 100.0, "voltage_v": 16.0, "dielectric": "X7R"},
    ),
    ReviewedPart(
        identity="grm188r61a106ke69d",
        family="capacitor-0603",
        package="Murata 0603 (1608 metric) X5R MLCC",
        bundle="kicad-standard",
        symbol="Device:C",
        footprint="Capacitor_SMD:C_0603_1608Metric",
        physical_features=frozenset({"capacitor-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://www.murata.com/en-global/products/productdetail?partno=GRM188R61A106KE69D",
        ),
        lcsc="C77044",
        operating_limits={"capacitance_uf": 10.0, "voltage_v": 10.0, "dielectric": "X5R"},
    ),
    ReviewedPart(
        identity="erj-3ekf1002v",
        family="resistor-0603",
        package="Panasonic 0603 (1608 metric) thick-film resistor",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric",
        physical_features=frozenset({"resistor-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://industrial.panasonic.com/cdbs/www-data/pdf/RDM0000/AOA0000C304.pdf",
        ),
        # No stock LCSC code is reviewed for this exact order code; the
        # Panasonic document above is the only reviewed source.
        operating_limits={"resistance_ohm": 10000.0, "tolerance_fraction": 0.01},
    ),
    # --- Reviewed interface, module, relay, and regulator parts (2026-09-16) --
    # Each record below names an exact order code, the exact library pair a
    # design must emit verbatim, and the manufacturer document it was reviewed
    # against.  Bundle-backed records take their MPN, symbol, and footprint
    # names from that bundle's own manifest ("<bundle>:<manifest name>");
    # kicad-standard records use the stock KiCad library ids unchanged, exactly
    # as the 132289 and stm32l031k6t6 entries do.  Nothing here is a
    # substitution claim: a record is usable only with its own pair.
    ReviewedPart(
        identity="max31855kasa+",
        family="max31855",
        package="Analog Devices MAX31855KASA+ SOIC-8, 3.9 x 4.9 mm, 1.27 mm pitch",
        bundle="kicad-standard",
        symbol="Sensor_Temperature:MAX31855KASA",
        footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
        physical_features=frozenset({"thermocouple-converter", "spi-sensor"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=(
            "https://www.analog.com/media/en/technical-documentation/data-sheets/MAX31855.pdf",
        ),
        operating_limits={"supply_min_v": 3.0, "supply_max_v": 3.6},
        port_pins={
            "ground": "1",
            "t_minus": "2",
            "t_plus": "3",
            "vcc": "4",
            "sck": "5",
            "chip_select": "6",
            "so": "7",
            "no_connect": "8",
        },
    ),
    # MAX485ESA+ and MAX485ESA+T are the same 8-lead narrow SO device from the
    # ADI ordering table; the trailing +T is the tape-and-reel shipping carrier,
    # not a different package or grade (see the membership comment below).
    ReviewedPart(
        identity="max485esa+",
        family="max485",
        package="Analog Devices MAX485ESA+ SOIC-8, 8-lead narrow, 3.9 x 4.9 mm, 1.27 mm pitch",
        bundle="kicad-standard",
        symbol="Interface_UART:MAX485E",
        footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
        physical_features=frozenset({"rs485-transceiver"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=(
            "https://www.analog.com/media/en/technical-documentation/data-sheets/MAX1487-MAX491.pdf",
            "https://www.analog.com/en/products/max485.html",
        ),
        operating_limits={"supply_min_v": 4.75, "supply_max_v": 5.25},
        port_pins={
            "ro": "1",
            "re_n": "2",
            "de": "3",
            "di": "4",
            "ground": "5",
            "a": "6",
            "b": "7",
            "vcc": "8",
        },
    ),
    ReviewedPart(
        identity="max485esa+t",
        family="max485",
        package="Analog Devices MAX485ESA+T SOIC-8, 8-lead narrow, 3.9 x 4.9 mm, 1.27 mm pitch; +T is the tape-and-reel carrier for the MAX485ESA+ device",
        bundle="kicad-standard",
        symbol="Interface_UART:MAX485E",
        footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
        physical_features=frozenset({"rs485-transceiver"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=(
            "https://www.analog.com/media/en/technical-documentation/data-sheets/MAX1487-MAX491.pdf",
            "https://www.analog.com/en/products/max485.html",
        ),
        operating_limits={"supply_min_v": 4.75, "supply_max_v": 5.25},
        port_pins={
            "ro": "1",
            "re_n": "2",
            "de": "3",
            "di": "4",
            "ground": "5",
            "a": "6",
            "b": "7",
            "vcc": "8",
        },
    ),
    ReviewedPart(
        identity="abm8-272-t3",
        family="crystal-3225",
        package="Abracon ABM8-272-T3 12 MHz, 10 pF load, 3.2 x 2.5 mm four-pad crystal",
        bundle="kicad-standard",
        symbol="Device:Crystal_GND24",
        footprint="Crystal:Crystal_SMD_3225-4Pin_3.2x2.5mm",
        physical_features=frozenset({"crystal", "crystal-3225"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=("https://abracon.com/datasheets/ABM8-272-T3.pdf",),
    ),
    ReviewedPart(
        identity="x32258msb4si",
        family="crystal-3225",
        package="YXC X32258MSB4SI 8 MHz, 3.2 x 2.5 mm four-pad crystal",
        bundle="crystal-8mhz-3225",
        symbol="crystal-8mhz-3225:X32258MSB4SI",
        footprint="crystal-8mhz-3225:CRYSTAL-SMD_4P-L3.2-W2.5-BL",
        physical_features=frozenset({"crystal", "crystal-3225"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C2682774.pdf",),
        lcsc="C2682774",
    ),
    ReviewedPart(
        identity="w25q16jvss",
        family="spi-flash",
        package="Winbond W25Q16JV, 16 Mbit, SS 208-mil SOIC-8, 5.3 x 5.3 mm",
        bundle="kicad-standard",
        symbol="Memory_Flash:W25Q16JVSS",
        footprint="Package_SO:SOIC-8_5.3x5.3mm_P1.27mm",
        physical_features=frozenset({"flash-memory"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=(
            "https://www.winbond.com/hq/support/documentation/levelOne.jsp?__locale=en&DocNo=DA00-W25Q16JV.1",
        ),
    ),
    ReviewedPart(
        identity="rp2040",
        family="rp2040",
        package="Raspberry Pi RP2040 QFN-56, 7 x 7 mm, 0.40 mm pitch, 3.2 x 3.2 mm exposed pad",
        bundle="kicad-standard",
        symbol="MCU_RaspberryPi:RP2040",
        footprint="Package_DFN_QFN:QFN-56-1EP_7x7mm_P0.4mm_EP3.2x3.2mm",
        physical_features=frozenset({"microcontroller"}),
        contacts=tuple(str(number) for number in range(1, 58)),
        manufacturer_sources=("https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf",),
        operating_limits={
            "io_supply_min_v": 1.8,
            "io_supply_max_v": 3.3,
            "maximum_cpu_mhz": 133,
        },
        port_pins={
            "xin": "20",
            "xout": "21",
            "swclk": "24",
            "swdio": "25",
            "run": "26",
            "vreg_vin": "44",
            "vreg_vout": "45",
            "usb_dm": "46",
            "usb_dp": "47",
            "qspi_sclk": "52",
            "qspi_ss": "56",
            "ground": "57",
        },
    ),
    ReviewedPart(
        identity="sn65hvd230",
        family="sn65hvd230",
        package="Texas Instruments SN65HVD230 SOIC-8, 3.9 x 4.9 mm, 1.27 mm pitch",
        bundle="kicad-standard",
        symbol="Interface_CAN_LIN:SN65HVD230",
        footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
        physical_features=frozenset({"can-transceiver"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=("https://www.ti.com/lit/ds/symlink/sn65hvd230.pdf",),
        operating_limits={"supply_min_v": 3.0, "supply_max_v": 3.6, "data_rate_mbps": 1.0},
        port_pins={
            "txd": "1",
            "ground": "2",
            "vcc": "3",
            "rxd": "4",
            "vref": "5",
            "canl": "6",
            "canh": "7",
            "rs": "8",
        },
    ),
    # Espressif ESP32-S3-WROOM-1: N8R8 and N16R8 are the same 41-contact module
    # with the same pin map and the same 8 MB PSRAM, differing only in Quad-SPI
    # flash capacity (8 vs 16 MB).  They are separate reviewed identities (see
    # the membership comment below); neither stands in for the other silently.
    ReviewedPart(
        identity="esp32-s3-wroom-1-n8r8",
        family="esp32-s3",
        package="Espressif ESP32-S3-WROOM-1 41-contact SMD module, 18.0 x 25.5 x 3.1 mm, 3.0 to 3.6 V supply",
        bundle="esp32-s3-wroom-1",
        symbol="esp32-s3-wroom-1:ESP32-S3-WROOM-1",
        footprint="esp32-s3-wroom-1:WIRELM-SMD_ESP32-S3-WROOM-1",
        physical_features=frozenset({"microcontroller", "wifi-module", "bluetooth-le-soc"}),
        contacts=tuple(str(number) for number in range(1, 42)),
        manufacturer_sources=(
            "https://www.lcsc.com/datasheet/C2913201.pdf",
            "https://documentation.espressif.com/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf",
        ),
        lcsc="C2913201",
        operating_limits={
            "supply_min_v": 3.0,
            "supply_max_v": 3.6,
            "flash_mb": 8,
            "psram_mb": 8,
        },
        port_pins={
            "ground": "1",
            "vdd": "2",
            "enable": "3",
            "io0": "27",
            "io19": "13",
            "io20": "14",
            "rxd0": "36",
            "txd0": "37",
        },
    ),
    ReviewedPart(
        identity="esp32-s3-wroom-1-n16r8",
        family="esp32-s3",
        package="Espressif ESP32-S3-WROOM-1 41-contact SMD module, 18.0 x 25.5 x 3.1 mm, 3.0 to 3.6 V supply; 16 MB flash variant",
        bundle="esp32-s3-wroom-1-n16r8",
        symbol="esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1",
        footprint="esp32-s3-wroom-1-n16r8:WIRELM-SMD_ESP32-S3-WROOM-1",
        physical_features=frozenset({"microcontroller", "wifi-module", "bluetooth-le-soc"}),
        contacts=tuple(str(number) for number in range(1, 42)),
        manufacturer_sources=(
            "https://www.lcsc.com/datasheet/C2913202.pdf",
            "https://documentation.espressif.com/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf",
        ),
        lcsc="C2913202",
        operating_limits={
            "supply_min_v": 3.0,
            "supply_max_v": 3.6,
            "flash_mb": 16,
            "psram_mb": 8,
        },
        port_pins={
            "ground": "1",
            "vdd": "2",
            "enable": "3",
            "io0": "27",
            "io19": "13",
            "io20": "14",
            "rxd0": "36",
            "txd0": "37",
        },
    ),
    ReviewedPart(
        identity="esp32-c3-mini-1-n4",
        family="esp32-c3",
        package="Espressif ESP32-C3-MINI-1 53-contact SMD module, 13.2 x 16.6 x 2.4 mm, 3.0 to 3.6 V supply, 4 MB flash",
        bundle="esp32-c3-mini-1-n4",
        symbol="esp32-c3-mini-1-n4:ESP32-C3-MINI-1-N4",
        footprint="esp32-c3-mini-1-n4:WIFIM-SMD_ESP32-C3-MINI-1",
        physical_features=frozenset({"microcontroller", "wifi-module", "bluetooth-le-soc"}),
        contacts=tuple(str(number) for number in range(1, 54)),
        manufacturer_sources=(
            "https://www.espressif.com/sites/default/files/documentation/esp32-c3-mini-1_datasheet_en.pdf",
        ),
        lcsc="C2838502",
        operating_limits={
            "supply_min_v": 3.0,
            "supply_max_v": 3.6,
            "flash_mb": 4,
        },
        port_pins={
            "ground": "1",
            "vdd": "3",
            "enable": "8",
            "io0": "12",
            "io1": "13",
            "io18": "26",
            "io19": "27",
            "rxd0": "30",
            "txd0": "31",
        },
    ),
    ReviewedPart(
        identity="mcp23017t-e/ss",
        family="mcp23017",
        package="Microchip MCP23017T-E/SS SSOP-28, 10.2 x 5.3 mm, 0.65 mm pitch",
        bundle="mcp23017t-e-ss",
        symbol="mcp23017t-e-ss:MCP23017T-E_SS",
        footprint="mcp23017t-e-ss:SSOP-28_L10.2-W5.3-P0.65-LS7.8-BL",
        physical_features=frozenset({"io-expander", "i2c-gpio-expander"}),
        contacts=tuple(str(number) for number in range(1, 29)),
        manufacturer_sources=(
            "https://ww1.microchip.com/downloads/en/DeviceDoc/21952a.pdf",
            "https://www.lcsc.com/datasheet/C558584.pdf",
        ),
        lcsc="C558584",
        operating_limits={"supply_min_v": 1.8, "supply_max_v": 5.5, "gpio_count": 16},
        port_pins={
            "gpio_b0": "1",
            "gpio_b7": "8",
            "vdd": "9",
            "vss": "10",
            "scl": "12",
            "sda": "13",
            "a0": "15",
            "a1": "16",
            "a2": "17",
            "reset": "18",
            "gpio_a0": "21",
            "gpio_a7": "28",
        },
    ),
    ReviewedPart(
        identity="pca9685pw,118",
        family="pca9685",
        package="NXP PCA9685PW,118 TSSOP-28, 9.7 x 4.4 mm, 0.65 mm pitch",
        bundle="pca9685",
        symbol="pca9685:PCA9685PW118",
        footprint="pca9685:TSSOP-28_L9.7-W4.4-P0.65-LS6.4-TL",
        physical_features=frozenset({"pwm-driver", "i2c-pwm-driver"}),
        contacts=tuple(str(number) for number in range(1, 29)),
        manufacturer_sources=(
            "https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf",
            "https://www.lcsc.com/datasheet/C2678753.pdf",
        ),
        lcsc="C2678753",
        operating_limits={
            "supply_min_v": 2.3,
            "supply_max_v": 5.5,
            "pwm_channels": 16,
            "pwm_resolution_bits": 12,
        },
        port_pins={
            "a0": "1",
            "a5": "24",
            "output_enable": "23",
            "scl": "26",
            "sda": "27",
            "vdd": "28",
            "vss": "14",
        },
    ),
    ReviewedPart(
        identity="ads1115idgsr",
        family="ads1115",
        package="Texas Instruments ADS1115IDGSR VSSOP-10 (MSOP-10 footprint), 3.0 x 3.0 mm, 0.50 mm pitch",
        bundle="ads1115",
        symbol="ads1115:ADS1115IDGSR",
        footprint="ads1115:MSOP-10_L3.0-W3.0-P0.50-LS5.0-BL",
        physical_features=frozenset({"adc", "i2c-adc"}),
        contacts=tuple(str(number) for number in range(1, 11)),
        manufacturer_sources=(
            "https://www.ti.com/lit/ds/symlink/ads1115.pdf",
            "https://lcsc.com/product-detail/Analog-To-Digital-Converters-ADCs_TI_ADS1115IDGSR_ADS1115IDGSR_C37593.html",
        ),
        lcsc="C37593",
        operating_limits={
            "supply_min_v": 2.0,
            "supply_max_v": 5.5,
            "resolution_bits": 16,
            "single_ended_channels": 4,
        },
        port_pins={
            "addr": "1",
            "alert": "2",
            "ground": "3",
            "ain0": "4",
            "ain1": "5",
            "ain2": "6",
            "ain3": "7",
            "vdd": "8",
            "sda": "9",
            "scl": "10",
        },
    ),
    ReviewedPart(
        identity="drv8833pwpr",
        family="drv8833",
        package="Texas Instruments DRV8833PWPR TSSOP-16 with exposed pad, 5.0 x 4.4 mm, 0.65 mm pitch",
        bundle="drv8833",
        symbol="drv8833:DRV8833PWP",
        footprint="drv8833:TSSOP-16_L5.0-W4.4-P0.65-LS6.4-BL-EP",
        physical_features=frozenset({"motor-driver", "dual-h-bridge"}),
        contacts=tuple(str(number) for number in range(1, 18)),
        manufacturer_sources=(
            "https://www.ti.com/lit/ds/symlink/drv8833.pdf",
            "https://lcsc.com/product-detail/Motor-Drivers_TI_DRV8833PWP_DRV8833PWP_C50506.html",
        ),
        lcsc="C50506",
        operating_limits={
            "motor_supply_min_v": 2.7,
            "motor_supply_max_v": 10.8,
            "continuous_current_per_channel_a": 1.5,
            "peak_current_per_channel_a": 2.0,
        },
        port_pins={
            "sleep_n": "1",
            "aout1": "2",
            "aisense": "3",
            "aout2": "4",
            "bout2": "5",
            "bisense": "6",
            "bout1": "7",
            "fault_n": "8",
            "bin1": "9",
            "bin2": "10",
            "vcp": "11",
            "vm": "12",
            "ground": "13",
            "vint": "14",
            "ain2": "15",
            "ain1": "16",
            "exposed_pad": "17",
        },
    ),
    ReviewedPart(
        identity="srd-05vdc-sl-c",
        family="relay",
        package="Songle SRD-05VDC-SL-C sealed through-hole power relay, 19.0 x 15.2 x 15.5 mm",
        bundle="srd-05vdc-sl-c",
        symbol="srd-05vdc-sl-c:SRD-05VDC-SL-C",
        footprint="srd-05vdc-sl-c:RELAY-TH_SRD-XXVDC-XL-C",
        physical_features=frozenset({"relay", "through-hole-relay"}),
        contacts=("1", "2", "3", "4", "5"),
        manufacturer_sources=(
            "https://lcsc.com/product-detail/Relays_SRD-05VDC-SL-C_C35449.html",
            "https://www.songlerelay.com/public/uploads/product_pdf/20200914/5f5ec4b110c1e.pdf",
        ),
        lcsc="C35449",
        operating_limits={
            "coil_voltage_v": 5.0,
            "contact_current_a": 10.0,
            "contact_voltage_vac": 250.0,
            "contact_voltage_vdc": 30.0,
            "contact_form": "SPDT",
        },
    ),
    ReviewedPart(
        identity="uln2003adr",
        family="uln2003",
        package="Texas Instruments ULN2003ADR SOIC-16, 9.9 x 3.9 mm, 1.27 mm pitch",
        bundle="uln2003adr",
        symbol="uln2003adr:ULN2003ADR",
        footprint="uln2003adr:SOIC-16_L9.9-W3.9-P1.27-LS6.0-BL",
        physical_features=frozenset({"darlington-array"}),
        contacts=tuple(str(number) for number in range(1, 17)),
        manufacturer_sources=(
            "https://www.ti.com/lit/ds/symlink/uln2003a.pdf",
            "https://lcsc.com/product-detail/High-CurrentDrivers_TI_ULN2003ADR_ULN2003ADR_C7512.html",
        ),
        lcsc="C7512",
        operating_limits={
            "output_voltage_max_v": 50.0,
            "continuous_output_current_a": 0.5,
            "channels": 7,
        },
        port_pins={
            "in1": "1",
            "in7": "7",
            "ground": "8",
            "com": "9",
            "out7": "10",
            "out1": "16",
        },
    ),
    # Reviewed SMT 3.3 V regulators.  The keys below are the LDO's/converter's
    # own reviewed ratings; they deliberately do not use the "vin_max_v" /
    # "continuous_output_a" keys, which the evidence pass reads as the power
    # path's design envelope rather than one small regulator's absolute limit.
    ReviewedPart(
        identity="me6211c33m5g-n",
        family="linear-regulator",
        package="Microne ME6211C33M5G-N SOT-23-5, 3.0 x 1.7 mm, 0.95 mm pitch",
        bundle="me6211c33",
        symbol="me6211c33:ME6211C33M5G-N",
        footprint="me6211c33:SOT-23-5_L3.0-W1.7-P0.95-LS2.8-BL",
        physical_features=frozenset({"smt-3v3-regulator", "voltage-regulator", "ldo-regulator"}),
        contacts=("1", "2", "3", "4", "5"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C82942.pdf",),
        lcsc="C82942",
        operating_limits={
            "input_voltage_min_v": 2.5,
            "input_voltage_max_v": 6.0,
            "output_voltage_v": 3.3,
            "output_current_a": 0.5,
        },
        port_pins={
            "input": "VIN",
            "ground": "VSS",
            "enable": "CE",
            "no_connect": "NC",
            "output": "VOUT",
        },
        # The reviewed source-to-load path §9.39 proves: this linear regulator conducts
        # VIN to VOUT. Its siblings (`ap2112k-3.3trg1`, `ams1117-5.0`) already carry this
        # fact; without it a 5 V to 3.3 V shield has no transfer witness at all and the
        # wiring stage refuses with "no reviewed source-to-load transfer", even though the
        # part is the reviewed one its recipe emits. Pin names come from the record's own
        # `port_pins` above (Microne ME6211 datasheet: VIN pin 1, VOUT pin 5, SOT-23-5).
        power_transfer={"from_pin": "VIN", "to_pin": "VOUT"},
    ),
    ReviewedPart(
        identity="ap2112k-3.3trg1",
        family="linear-regulator",
        package="Diodes Incorporated AP2112K-3.3TRG1 SOT-25, 2.9 x 1.6 mm, 0.95 mm pitch",
        bundle="ap2112k-3v3",
        symbol="ap2112k-3v3:AP2112K-3.3TRG1",
        footprint="ap2112k-3v3:SOT-25-5_L2.9-W1.6-P0.95-LS2.8-BL",
        physical_features=frozenset({"smt-3v3-regulator", "voltage-regulator", "ldo-regulator"}),
        contacts=("1", "2", "3", "4", "5"),
        manufacturer_sources=(
            "https://www.diodes.com/assets/Datasheets/AP2112.pdf",
            "https://lcsc.com/product-detail/Low-Dropout-Regulators-LDO_DIODES_AP2112K-3-3TRG1_AP2112K-3-3TRG1_C51118.html",
        ),
        lcsc="C51118",
        operating_limits={
            "input_voltage_min_v": 2.5,
            "input_voltage_max_v": 6.0,
            "output_voltage_v": 3.3,
            "output_current_a": 0.6,
        },
        port_pins={
            "input": "VIN",
            "ground": "GND",
            "enable": "EN",
            "no_connect": "NC",
            "output": "VOUT",
        },
        # An LDO's source-to-load path is its own datasheet-documented VIN->VOUT
        # transfer; without it §9.39 has no reviewed path for a 5 V -> 3.3 V rail.
        power_transfer={"from_pin": "VIN", "to_pin": "VOUT"},
    ),
    ReviewedPart(
        identity="tlv62569dbvr",
        family="buck-regulator",
        package="Texas Instruments TLV62569DBVR SOT-23-5, 3.0 x 1.7 mm, 0.95 mm pitch",
        bundle="tlv62569",
        symbol="tlv62569:TLV62569DBVR",
        footprint="tlv62569:SOT-23-5_L3.0-W1.7-P0.95-LS2.8-BR",
        physical_features=frozenset({"smt-3v3-regulator", "voltage-regulator", "buck-regulator"}),
        contacts=("1", "2", "3", "4", "5"),
        manufacturer_sources=(
            "https://www.ti.com/lit/ds/symlink/tlv62569.pdf",
            "https://lcsc.com/product-detail/DC-DC-Converters_Texas-Instruments_TLV62569DBVR_Texas-Instruments-TI-TLV62569DBVR_C141836.html",
        ),
        lcsc="C141836",
        operating_limits={
            # §9.38 compares a typed VIN rail against the reviewed range via the
            # canonical vin_min_v/vin_max_v keys, so a power-transfer device must
            # declare them; the input_voltage_* spelling is kept for the
            # design-envelope pass.
            "vin_min_v": 2.5,
            "vin_max_v": 5.5,
            "input_voltage_min_v": 2.5,
            "input_voltage_max_v": 5.5,
            "output_current_a": 2.0,
            "output_voltage_v": 3.3,
        },
        power_transfer={"from_pin": "VIN", "to_pin": "SW"},
        port_pins={
            "enable": "EN",
            "ground": "GND",
            "switch": "SW",
            "input": "VIN",
            "feedback": "FB",
        },
    ),
    # The fixed-output synchronous bucks the corpus's higher-voltage briefs resolve to
    # (`ap63203-3v3@1` for 3V3, `ap63205-5v@1` for 5V).  Both are rows of the same Diodes
    # datasheet and the same TSOT-26 pin table (FB 1, EN 2, VIN 3, GND 4, SW 5, BST 6),
    # and both are catalog parts (C780769 / C2071056), so each order code carries its own
    # record -- neither is inferred from the other, and the 5 V part deliberately does NOT
    # carry the 3.3 V class.
    #
    # The §9.39 fact is the converter's own datasheet path: the high-side FET conducts VIN
    # to the switch node SW, and the recipe's own power inductor (a plain `L` part from
    # switch to output) carries it on to the regulated rail -- the generic inductor edge
    # `_reviewed_transfer_edges` already adds.  Without these records the group's MPN
    # resolves to no reviewed identity at all, so an 18 V -> 3V3 brief had no
    # source-to-load witness and the wiring stage refused with
    # "E_POWER_TRANSFER 'regulator': no reviewed source-to-load transfer from '+18V' to
    # '+3V3'" even though the part the recipe emitted is the reviewed one.
    #
    # The bootstrap specification is the part's mandatory external, straight from the
    # datasheet's typical application (BST to SW, 0.1 uF), and the bundle's own
    # `watch_out_for` note.  §9.37 then proves the recipe's own bootstrap capacitor.
    ReviewedPart(
        identity="ap63203wu-7",
        family="buck-converter",
        package=(
            "Diodes AP63203WU-7 fixed-3.3 V synchronous buck, TSOT-26, "
            "3.8-32 V input, 2 A"
        ),
        bundle="ap63203",
        symbol="ap63203:AP63203WU-7",
        footprint="ap63203:TSOT-26_L2.9-W1.6-P0.95-LS2.8-BL",
        physical_features=frozenset(
            {"buck-converter", "buck-regulator", "voltage-regulator", "smt-3v3-regulator"}
        ),
        function_keys=frozenset({"fixed-3v3-buck"}),
        contacts=("1", "2", "3", "4", "5", "6"),
        manufacturer_sources=(
            "https://www.diodes.com/assets/Datasheets/AP63200-AP63201-AP63203-AP63205.pdf",
            "https://lcsc.com/product-detail/DC-DC-Converters_DIODES_AP63203WU-7_AP63203WU-7_C780769.html",
        ),
        lcsc="C780769",
        operating_limits={
            "vin_min_v": 3.8,
            "vin_max_v": 32.0,
            "input_voltage_min_v": 3.8,
            "input_voltage_max_v": 32.0,
            "output_voltage_v": 3.3,
            "output_current_a": 2.0,
        },
        bootstrap={"positive_pin": "BST", "negative_pin": "SW", "capacitance_uf": 0.1},
        power_transfer={"from_pin": "VIN", "to_pin": "SW"},
        port_pins={
            "input": "VIN",
            "ground": "GND",
            "switch": "SW",
            "feedback": "FB",
            "enable": "EN",
            "bootstrap": "BST",
        },
    ),
    ReviewedPart(
        identity="ap63205wu-7",
        family="buck-converter",
        package=(
            "Diodes AP63205WU-7 fixed-5 V synchronous buck, TSOT-26, "
            "3.8-32 V input, 2 A"
        ),
        bundle="ap63205",
        symbol="ap63205:AP63205WU-7",
        footprint="ap63205:TSOT-23-6_L2.9-W1.6-P0.95-LS2.8-BL",
        physical_features=frozenset({"buck-converter", "buck-regulator", "voltage-regulator"}),
        function_keys=frozenset({"fixed-5v-buck"}),
        contacts=("1", "2", "3", "4", "5", "6"),
        manufacturer_sources=(
            "https://www.diodes.com/assets/Datasheets/AP63200-AP63201-AP63203-AP63205.pdf",
            "https://lcsc.com/product-detail/DC-DC-Converters_DIODES_AP63205WU-7_AP63205WU-7_C2071056.html",
        ),
        lcsc="C2071056",
        operating_limits={
            "vin_min_v": 3.8,
            "vin_max_v": 32.0,
            "input_voltage_min_v": 3.8,
            "input_voltage_max_v": 32.0,
            "output_voltage_v": 5.0,
            "output_current_a": 2.0,
        },
        bootstrap={"positive_pin": "BST", "negative_pin": "SW", "capacitance_uf": 0.1},
        power_transfer={"from_pin": "VIN", "to_pin": "SW"},
        port_pins={
            "input": "VIN",
            "ground": "GND",
            "switch": "SW",
            "feedback": "FB",
            "enable": "EN",
            "bootstrap": "BST",
        },
    ),
    ReviewedPart(
        identity="u-a-24ss-w-2",
        family="usb-a-receptacle",
        package="Korean Hroparts U-A-24SS-W-2 USB 2.0 type-A receptacle, right-angle SMD, 1.5 A",
        bundle="usb-a-24ss-w-2",
        symbol="usb-a-24ss-w-2:U-A-24SS-W-2",
        footprint="usb-a-24ss-w-2:USB-A-SMD_U-A-24SS-W-2",
        physical_features=frozenset({"usb-a-receptacle", "usb-connector"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=(
            "https://lcsc.com/product-detail/USB-Connectors_Korean-Hroparts-Elec-U-A-24SS-W-2_C530629.html",
        ),
        lcsc="C530629",
        operating_limits={"voltage_v": 5.0, "current_a": 1.5, "usb_generation": "2.0"},
        port_pins={"vbus": "1", "d_minus": "2", "d_plus": "3", "ground": "4"},
        support_network={
            "shell_pads": ("5", "6"),
            "shell_pads_must_be_grounded": True,
        },
    ),
    # --- Reviewed recipe and package-identity completions (2026-09-16) -------
    # These close the "recipe emits an MPN nothing resolves" gap: every identity
    # below is either an order code a registered recipe already emits or a
    # documented order code from the reference evidence, and each is paired with
    # the exact symbol/footprint that recipe or reference row declares.
    ReviewedPart(
        identity="attiny1614-ssnr",
        family="attiny1614",
        package="Microchip ATtiny1614 SOIC-14 (SOIC150), 3.9 x 8.7 mm, 1.27 mm pitch",
        bundle="kicad-standard",
        symbol="MCU_Microchip_ATtiny:ATtiny1614-SS",
        footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm",
        physical_features=frozenset({"microcontroller", "updi-programmable", "attiny1614"}),
        contacts=tuple(str(number) for number in range(1, 15)),
        manufacturer_sources=(
            "https://ww1.microchip.com/downloads/en/DeviceDoc/ATtiny1614-16-17-DataSheet-DS40002204A.pdf",
            "https://ww1.microchip.com/downloads/aemDocuments/documents/MCU08/ProductDocuments/DataSheets/ATtiny1614-16-17-DataSheet-DS40002204.pdf",
        ),
        operating_limits={
            "supply_min_v": 1.8,
            "supply_max_v": 5.5,
            "temperature_min_c": -40,
            "temperature_max_c": 105,
            "maximum_cpu_mhz": 20,
        },
    ),
    ReviewedPart(
        identity="attiny412-ssn",
        family="attiny412",
        package="Microchip ATtiny412 SOIC-8 (SOIC150), 3.9 x 4.9 mm, 1.27 mm pitch",
        bundle="kicad-standard",
        symbol="MCU_Microchip_ATtiny:ATtiny412-SS",
        footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
        physical_features=frozenset({"microcontroller", "updi-programmable", "attiny412"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=(
            "https://ww1.microchip.com/downloads/aemDocuments/documents/MCU08/ProductDocuments/DataSheets/ATtiny212-214-412-414-416-DataSheet-DS40002287A.pdf",
        ),
        operating_limits={
            "supply_min_v": 1.8,
            "supply_max_v": 5.5,
            "temperature_min_c": -40,
            "temperature_max_c": 105,
            "maximum_cpu_mhz": 20,
        },
    ),
    # Two of the six MCUs the Surprise-me brief generator names (`BRIEF_SLOTS["mcu"]`) had no
    # reviewed carrier, while the class they answer (`microcontroller`) is covered and therefore
    # requires one: a brief that drew either died at the BOM with "requires 1 real
    # microcontroller, found 0" whatever the unit emitted (live run KC-YMEWZV, seed 28 —
    # ATtiny1604 — where the unit offered the right ordering code three times and no resolvable
    # symbol/footprint pair, so no retry could ever satisfy the demand). Both records are the
    # standard KiCad symbol/footprint pair plus the ordering code the offline catalog resolves;
    # `tests/test_onboarding_examples.py` holds every generator MCU to this.
    ReviewedPart(
        identity="attiny1604-ssn",
        family="attiny1604",
        package="Microchip ATtiny1604 SOIC-14 (SOIC150), 3.9 x 8.7 mm, 1.27 mm pitch",
        bundle="kicad-standard",
        symbol="MCU_Microchip_ATtiny:ATtiny1604-SS",
        footprint="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm",
        physical_features=frozenset({"microcontroller", "updi-programmable", "attiny1604"}),
        contacts=tuple(str(number) for number in range(1, 15)),
        manufacturer_sources=(
            "https://www.microchip.com/en-us/product/ATtiny1604",
            "https://lcsc.com/product-detail/C614830.html",
        ),
        lcsc="C614830",
        operating_limits={
            "supply_min_v": 1.8,
            "supply_max_v": 5.5,
            "temperature_min_c": -40,
            "temperature_max_c": 105,
            "maximum_cpu_mhz": 20,
        },
    ),
    ReviewedPart(
        identity="stm32g030f6p6",
        family="stm32g0",
        package="STMicro STM32G030F6P6 TSSOP-20, 4.4 x 6.5 mm, 0.65 mm pitch",
        bundle="kicad-standard",
        symbol="MCU_ST_STM32G0:STM32G030F6Px",
        footprint="Package_SO:TSSOP-20_4.4x6.5mm_P0.65mm",
        physical_features=frozenset({"microcontroller", "swd-programmable", "stm32g0"}),
        contacts=tuple(str(number) for number in range(1, 21)),
        manufacturer_sources=(
            "https://www.st.com/en/microcontrollers-microprocessors/stm32g030f6.html",
            "https://lcsc.com/product-detail/C724040.html",
        ),
        lcsc="C724040",
        operating_limits={
            "supply_min_v": 2.0,
            "supply_max_v": 3.6,
            "temperature_min_c": -40,
            "temperature_max_c": 85,
            "maximum_cpu_mhz": 64,
        },
    ),
    # Microchip DS21952A orders MCP23017-E/SO as the 28-lead SOIC (wide body).
    # That is a different package from the SSOP-28 MCP23017T-E/SS record above,
    # so neither is a member of the other's identity; each is selected only with
    # its own symbol/footprint pair.
    ReviewedPart(
        identity="mcp23017-e/so",
        family="mcp23017",
        package="Microchip MCP23017-E/SO SOIC-28W (wide body), 7.5 x 17.9 mm, 1.27 mm pitch",
        bundle="kicad-standard",
        symbol="Interface_Expansion:MCP23017_SO",
        footprint="Package_SO:SOIC-28W_7.5x17.9mm_P1.27mm",
        physical_features=frozenset({"io-expander", "i2c-gpio-expander"}),
        contacts=tuple(str(number) for number in range(1, 29)),
        manufacturer_sources=("https://ww1.microchip.com/downloads/en/DeviceDoc/21952a.pdf",),
        operating_limits={"supply_min_v": 1.8, "supply_max_v": 5.5, "gpio_count": 16},
        port_pins={
            "gpio_b0": "1",
            "gpio_b7": "8",
            "vdd": "9",
            "vss": "10",
            "scl": "12",
            "sda": "13",
            "a0": "15",
            "a1": "16",
            "a2": "17",
            "reset": "18",
            "gpio_a0": "21",
            "gpio_a7": "28",
        },
    ),
    # Amphenol's USB-C receptacle is documented under both the hyphen and the
    # '#2A' order-code spellings; the KiCad land pattern keeps the hyphen form.
    # Both records are the same part and the same pair, deliberately separate
    # identities so a design may assert either documented order code.
    ReviewedPart(
        identity="12401610e4#2a",
        family="usb-c-receptacle",
        package="Amphenol 12401610E4#2A (also documented as 12401610E4-2A) USB-C receptacle; the KiCad footprint below uses the hyphen spelling",
        bundle="kicad-standard",
        symbol="Connector:USB_C_Receptacle",
        footprint="Connector_USB:USB_C_Receptacle_Amphenol_12401610E4-2A",
        physical_features=frozenset({"usb-c-receptacle"}),
        contacts=(
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A11",
            "A12",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B9",
            "B10",
            "B11",
            "B12",
            "S1",
        ),
        manufacturer_sources=(
            "https://cdn.amphenol-cs.com/media/wysiwyg/files/drawing/c12401610_c.pdf",
        ),
    ),
    # Advanced Monolithic AMS1117-5.0: the isolated RS-485 node's field-side
    # 5 V regulator.  It is a 5 V part, so it MUST NOT carry the 3.3 V class.
    # The vendored easyeda bundle is the identity the parts loader resolves for
    # MPN AMS1117-5.0 (_curated_part_indexes indexes the vendored copy and the
    # user-wide fetch cache by MPN), so the reviewed record must name that same
    # symbol/footprint pair: physical_inventory_record only classifies a part
    # whose (mpn, symbol, footprint) all match the reviewed record exactly.
    ReviewedPart(
        identity="ams1117-5.0",
        family="linear-regulator",
        package="Advanced Monolithic Systems AMS1117-5.0 SOT-223, 3.5 x 6.5 mm, tab is pin 2",
        bundle="ams1117-5v0-fixed",
        symbol="ams1117-5v0-fixed:AMS1117-5.0",
        footprint="ams1117-5v0-fixed:SOT-223_L6.5-W3.5-P2.30-LS7.0-BR",
        physical_features=frozenset({"voltage-regulator", "linear-regulator", "ldo-regulator"}),
        contacts=("1", "2", "3"),
        manufacturer_sources=(
            "https://www.advanced-monolithic.com/pdf/ds1117.pdf",
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8756672206577377280",
        ),
        lcsc="C6187",
        operating_limits={
            # AMS1117 datasheet (Advanced Monolithic Systems DS1117): absolute
            # maximum input voltage 15 V, and the drop-out voltage (VIN-VOUT) is
            # guaranteed maximum 1.3 V at IOUT = 0.8 A.  A 5.0 V output therefore
            # needs at least 5.0 + 1.3 = 6.3 V in for guaranteed regulation; the
            # device conducts from its VIN pin to its VOUT pin (three-terminal
            # pass element), which is the reviewed source-to-load transfer §9.39
            # needs from a linear regulator.
            "input_voltage_min_v": 6.3,
            "input_voltage_max_v": 15.0,
            "output_voltage_v": 5.0,
            "output_current_a": 1.0,
        },
        power_transfer={"from_pin": "3", "to_pin": "2"},
        port_pins={"ground": "1", "output": "2", "input": "3"},
    ),
    # Kyocera AVX TAJB686K010RNJ tantalum, EIA 3528-21 case B.  The installed
    # KiCad library has no AVX-B land pattern; CP_EIA-3528-21_Kemet-B is the
    # installed 3528-21 B land pattern and is what the reference row now uses.
    ReviewedPart(
        identity="tajb686k010rnj",
        family="tantalum-capacitor",
        package="Kyocera AVX TAJ series 68 uF 10 V tantalum, EIA 3528-21 case B, 3.5 x 2.8 mm",
        bundle="kicad-standard",
        symbol="Device:C",
        footprint="Capacitor_Tantalum_SMD:CP_EIA-3528-21_Kemet-B",
        physical_features=frozenset({"tantalum-capacitor", "output-bulk-capacitor"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588880875023319040",
        ),
        lcsc="C111253",
        operating_limits={
            "capacitance_uf": 68.0,
            "voltage_v": 10.0,
            "tolerance_fraction": 0.1,
        },
    ),
    ReviewedPart(
        identity="rc1206fr-07680rl",
        family="resistor-1206",
        package="YAGEO RC series 1206 (3216 metric) thick-film resistor, 250 mW",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_1206_3216Metric",
        physical_features=frozenset({"resistor-1206"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C137240.pdf",),
        lcsc="C137240",
        operating_limits={
            "resistance_ohm": 680.0,
            "tolerance_fraction": 0.01,
            "power_w": 0.25,
        },
    ),
    ReviewedPart(
        identity="ltst-c190kgkt",
        family="led-0603",
        package="Lite-On LTST-C190KGKT 0603 (1608 metric) yellow-green indicator LED",
        bundle="kicad-standard",
        symbol="Device:LED",
        footprint="LED_SMD:LED_0603_1608Metric",
        physical_features=frozenset({"led-0603", "indicator-led"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588884651616583680",
        ),
        lcsc="C125094",
        operating_limits={"forward_voltage_v": 2.0, "forward_current_ma": 20.0},
        port_pins={"cathode": "1", "anode": "2"},
    ),
    # ALPS EC11E15244G1: through-hole incremental encoder with an integral push
    # switch.  KiCad 9 renamed the old Switch:SW_Encoder / Encoder_Alps pair to
    # Device:RotaryEncoder_Switch / Rotary_Encoder:RotaryEncoder_Alps_...; the
    # ids below are the installed ones.  Contacts are the five signal contacts
    # (A/B/C quadrature plus S1/S2 switch); pad MP is the frame/bushing tab.
    ReviewedPart(
        identity="ec11e15244g1",
        family="rotary-encoder",
        package="ALPSALPINE EC11E15244G1 vertical through-hole incremental rotary encoder with push switch, 20 mm body land pattern",
        bundle="ec11e15244g1",
        symbol="ec11e15244g1:EC11E15244G1",
        footprint="ec11e15244g1:SW-TH_EC11E15244G1",
        physical_features=frozenset({"rotary-encoder", "through-hole-rotary-encoder"}),
        contacts=("A", "B", "C", "D", "E", "6", "7"),
        manufacturer_sources=(
            "https://lcsc.com/product-detail/Others_ALPS-Electric-EC11E15244G1_C370970.html",
            "https://tech.alpsalpine.com/prod/e/html/encoder/incremental/ec11/ec11e15244g1.html",
        ),
        lcsc="C370970",
        port_pins={
            "quadrature_a": "A",
            "quadrature_b": "B",
            "common": "C",
            "switch_1": "D",
            "switch_2": "E",
        },
        # Pads 6/7 are the 4.0 x 2.3 mm oval frame/bushing tabs.  They are
        # mechanical: whether they are grounded is a board-level decision, so no
        # grounding claim is asserted here.
        support_network={"mechanical_pads": ("6", "7")},
    ),
    ReviewedPart(
        identity="mcp6001t-i/ot",
        family="operational-amplifier",
        package="Microchip MCP6001T-I/OT SOT-23-5, 3.0 x 1.7 mm, 1.6 V to 6.0 V single op-amp",
        bundle="mcp6001",
        symbol="mcp6001:MCP6001T-I_OT",
        footprint="mcp6001:SOT-23-5_L3.0-W1.7-P0.95-LS2.8-BR",
        physical_features=frozenset({"operational-amplifier", "single-operational-amplifier"}),
        contacts=("1", "2", "3", "4", "5"),
        manufacturer_sources=(
            "https://lcsc.com/product-detail/Low-Power-OpAmps_MICROCHIP_MCP6001T-I-OT_MCP6001T-I-OT_C116490.html",
        ),
        lcsc="C116490",
        operating_limits={
            "supply_min_v": 1.8,
            "supply_max_v": 6.0,
            "gain_bandwidth_mhz": 1.0,
            "quiescent_current_ua": 100.0,
        },
        port_pins={"output": "1", "vss": "2", "vin_plus": "3", "vin_minus": "4", "vdd": "5"},
    ),
    ReviewedPart(
        identity="ch224k",
        family="usb-pd-controller",
        package="WCH CH224K USB PD sink/trigger controller, ESSOP-10 with exposed pad",
        bundle="ch224k",
        symbol="ch224k:CH224K",
        footprint="ch224k:ESSOP-10_L4.9-W3.9-P1.0-LS6.0-TL-EP",
        physical_features=frozenset({"usb-pd-controller", "voltage-selector"}),
        contacts=("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C970725.pdf",),
        lcsc="C970725",
        operating_limits={
            # §9.38 reads the canonical vin_* keys; the WCH CH224K datasheet
            # specifies a 4.0-30 V input range (and the CFG table selects the
            # negotiated 5/9/12/15/20 V request).
            "vin_min_v": 4.0,
            "vin_max_v": 30.0,
            "input_voltage_min_v": 4.0,
            "input_voltage_max_v": 30.0,
            "output_power_w": 100.0,
        },
        port_pins={
            "vdd": "1",
            "cfg2": "2",
            "cfg3": "3",
            "usb_dp": "4",
            "usb_dm": "5",
            "cc2": "6",
            "cc1": "7",
            "vbus": "8",
            "cfg1": "9",
            "power_good": "10",
            "ground": "11",
        },
    ),
    ReviewedPart(
        identity="esp32-s3-mini-1-n8",
        family="esp32-s3-mini-1",
        package="Espressif ESP32-S3-MINI-1 61-contact SMD module, 15.8 x 20.3 mm, 3.0 to 3.6 V supply, 8 MB flash",
        bundle="esp32-s3-mini-1",
        symbol="esp32-s3-mini-1:ESP32-S3-MINI-1-N8",
        footprint="esp32-s3-mini-1:BULETM-SMD_ESP32-S3-MINI-1-N8",
        physical_features=frozenset({"microcontroller", "wifi-module", "bluetooth-le-soc"}),
        contacts=tuple(str(number) for number in range(1, 61)) + ("GND",),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C2913206.pdf",),
        lcsc="C2913206",
        operating_limits={"supply_min_v": 3.0, "supply_max_v": 3.6, "flash_mb": 8},
        port_pins={
            "ground": "1",
            "vdd": "3",
            "enable": "45",
            "io0": "4",
            "usb_dm": "23",
            "usb_dp": "24",
        },
    ),
    ReviewedPart(
        identity="usblc6-2sc6",
        family="usb-esd-protection",
        package="UMW USBLC6-2SC6 USB ESD protection array, SOT-23-6, 2.9 x 1.6 mm",
        bundle="usblc6-2sc6",
        symbol="usblc6-2sc6:USBLC6-2SC6_C2687116",
        footprint="usblc6-2sc6:SOT-23-6_L2.9-W1.6-P0.95-LS2.8-BL",
        physical_features=frozenset({"usb-esd-protection", "tvs-diode-array"}),
        contacts=("1", "2", "3", "4", "5", "6"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C2687116.pdf",),
        lcsc="C2687116",
        port_pins={"io1": "1", "ground": "2", "io2": "3", "vbus": "5"},
    ),
    ReviewedPart(
        identity="ws2812b-b/t",
        family="ws2812b",
        package="Worldsemi WS2812B 5050 addressable RGB LED, 5.0 x 5.0 mm, 4-pad",
        bundle="ws2812b",
        symbol="ws2812b:WS2812B-B_W",
        footprint="ws2812b:LED-SMD_4P-L5.0-W5.0-TL_WS2812B-B",
        physical_features=frozenset({"addressable-led", "ws2812b", "rgb-led"}),
        contacts=("1", "2", "3", "4"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C2761795.pdf",),
        lcsc="C2761795",
        operating_limits={"supply_min_v": 3.7, "supply_max_v": 5.3},
        port_pins={"vdd": "1", "dout": "2", "vss": "3", "din": "4"},
    ),
    ReviewedPart(
        identity="ch32v003j4m6",
        family="ch32v003",
        package="WCH CH32V003J4M6 SOP-8, 4.9 x 3.8 mm, 1.27 mm pitch",
        bundle="ch32v003j4m6",
        symbol="ch32v003j4m6:CH32V003J4M6",
        footprint="ch32v003j4m6:SOP-8_L4.9-W3.8-P1.27-LS6.0-BL",
        physical_features=frozenset({"microcontroller"}),
        contacts=tuple(str(number) for number in range(1, 9)),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C5346354.pdf",),
        lcsc="C5346354",
        operating_limits={"supply_min_v": 2.7, "supply_max_v": 5.5, "maximum_cpu_mhz": 48},
        port_pins={"vdd": "4", "vss": "2"},
    ),
    ReviewedPart(
        identity="a4988settr-t",
        family="a4988",
        package="Allegro A4988SETTR-T WQFN-28 with exposed pad, 5.0 x 5.0 mm, 0.50 mm pitch",
        bundle="a4988",
        symbol="a4988:A4988SETTR-T",
        footprint="a4988:WQFN-28_L5.0-W5.0-P0.50-BL-EP3.2",
        physical_features=frozenset({"stepper-driver"}),
        contacts=tuple(str(number) for number in range(1, 30)),
        manufacturer_sources=(
            "https://lcsc.com/product-detail/MOS-Drivers_ALLEGRO_A4988SETTR-T_A4988SETTR-T_C38437.html",
        ),
        lcsc="C38437",
        operating_limits={"motor_supply_max_v": 35.0, "output_current_a": 2.0},
        port_pins={
            "enable_n": "2",
            "ms1": "9",
            "ms2": "10",
            "ms3": "11",
            "reset_n": "12",
            "sleep_n": "14",
            "vdd": "15",
            "step": "16",
            "ref": "17",
            "dir": "19",
            "vbb1": "22",
            "vbb2": "28",
            "exposed_pad": "29",
        },
    ),
    ReviewedPart(
        identity="ch340c",
        family="usb-uart-bridge",
        package="WCH CH340C SOP-16, 10.0 x 3.9 mm, 1.27 mm pitch",
        bundle="ch340c",
        symbol="ch340c:CH340C",
        footprint="ch340c:SOP-16_L10.0-W3.9-P1.27-LS6.0-BL",
        physical_features=frozenset({"usb-uart-bridge"}),
        contacts=tuple(str(number) for number in range(1, 17)),
        manufacturer_sources=("https://lcsc.com/product-detail/USB_CH340C_C84681.html",),
        lcsc="C84681",
        operating_limits={"supply_min_v": 3.3, "supply_max_v": 5.5, "data_rate_mbps": 2.0},
        port_pins={
            "ground": "1",
            "txd": "2",
            "rxd": "3",
            "v3": "4",
            "usb_dp": "5",
            "usb_dm": "6",
            "vcc": "16",
        },
    ),
    ReviewedPart(
        identity="esp32-wroom-32e-n4",
        family="esp32-wroom-32e",
        package="Espressif ESP32-WROOM-32E 39-contact SMD module, 18.0 x 25.5 mm, 3.0 to 3.6 V supply, 4 MB flash",
        bundle="esp32-wroom-32e-n4",
        symbol="esp32-wroom-32e-n4:ESP32-WROOM-32E",
        footprint="esp32-wroom-32e-n4:WIFI-SMD_ESP32-WROOM-32E",
        physical_features=frozenset({"microcontroller", "wifi-module", "bluetooth-le-soc"}),
        contacts=tuple(str(number) for number in range(1, 40)),
        manufacturer_sources=(
            "https://lcsc.com/product-detail/WIFI-Modules_Espressif-Systems-ESP32-WROOM-32E-4MB_C701341.html",
        ),
        lcsc="C701341",
        operating_limits={"supply_min_v": 3.0, "supply_max_v": 3.6, "flash_mb": 4},
        port_pins={
            "ground": "1",
            "vdd": "2",
            "enable": "3",
            "io0": "25",
            "rxd0": "34",
            "txd0": "35",
        },
    ),
    ReviewedPart(
        identity="s2b-ph-sm4-tb(lf)(sn)",
        family="wire-to-board-connector",
        package="JST PH series 1x02, 2.00 mm-pitch side-entry SMD wire-to-board receptacle",
        bundle="jst-ph-2p",
        symbol="jst-ph-2p:S2B-PH-SM4-TB",
        footprint="jst-ph-2p:CONN-SMD_P2.00_S2B-PH-SM4-TB-LF-SN",
        physical_features=frozenset({"wire-to-board-connector", "power-connector", "jst-ph"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://lcsc.com/product-detail/_JST-Sales-America_S2B-PH-SM4-TB-LF-SN_JST-Sales-America-S2B-PH-SM4-TB-LF-SN_C295747.html",
        ),
        lcsc="C295747",
        operating_limits={"pitch_mm": 2.0, "positions": 2, "current_a": 2.0, "voltage_v": 100.0},
        support_network={
            "mounting_pads": ("3", "4"),
            "mounting_pads_must_be_soldered": True,
        },
    ),
)


# ------------------------------------------------------------- researched records
# A demanded class the vendored records cannot answer is researched once and stored as data
# (`kicraft.parts_library.researched_records`), never as another literal in this file: that is
# what makes it a one-time cost for every later run and every other project on the machine. The
# merge is cached on the file's own state, so a record added mid-run is visible to the very next
# check in the same process -- which is what lets one stage research a part and the next select it.
_RESEARCHED_CACHE: tuple[tuple[object, ...], tuple[ReviewedPart, ...]] | None = None


def _researched_parts() -> tuple[ReviewedPart, ...]:
    """Records this machine researched for itself, cached per state of the record file."""
    global _RESEARCHED_CACHE
    from kicraft.parts_library import researched_records

    path = researched_records.records_path()
    try:
        stat = path.stat()
        stamp: tuple[object, ...] = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = (str(path), None, None)
    if _RESEARCHED_CACHE is not None and _RESEARCHED_CACHE[0] == stamp:
        return _RESEARCHED_CACHE[1]

    rows: list[ReviewedPart] = []
    for row in researched_records.load(path):
        try:
            rows.append(_reviewed_part_from_record(row))
        except (TypeError, ValueError):
            # A hand-edited row that cannot be read must not take the pipeline down: it is
            # skipped, and the demand it was meant to answer is researched again on demand.
            continue
    _RESEARCHED_CACHE = (stamp, tuple(rows))
    return _RESEARCHED_CACHE[1]


def _reviewed_part_from_record(row: Mapping[str, object]) -> ReviewedPart:
    """One stored record as a ReviewedPart. Raises ValueError when it names no part or class."""

    def _texts(key: str) -> tuple[str, ...]:
        value = row.get(key)
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            return (value.decode() if isinstance(value, bytes) else value,)
        return tuple(str(item) for item in value)  # type: ignore[union-attr]

    def _mapping(key: str) -> dict:
        value = row.get(key)
        return dict(value) if isinstance(value, Mapping) else {}

    identity = str(row.get("identity") or "").strip().casefold()
    family = str(row.get("family") or "").strip().casefold()
    if not identity or not family:
        raise ValueError("a researched record needs an identity and a family")
    bundle = row.get("bundle")
    symbol = row.get("symbol")
    footprint = row.get("footprint")
    return ReviewedPart(
        identity=identity,
        family=family,
        package=str(row.get("package") or ""),
        bundle=None if bundle is None else str(bundle),
        symbol=None if symbol is None else str(symbol),
        footprint=None if footprint is None else str(footprint),
        physical_features=frozenset(_texts("physical_features")),
        function_keys=frozenset(_texts("function_keys")),
        contacts=_texts("contacts"),
        manufacturer_sources=_texts("manufacturer_sources"),
        lcsc=None if row.get("lcsc") is None else str(row["lcsc"]),
        current_feedback=_mapping("current_feedback"),
        power_transfer=_mapping("power_transfer"),
        operating_limits=_mapping("operating_limits"),
        bootstrap=_mapping("bootstrap"),
        port_pins={str(key): str(value) for key, value in _mapping("port_pins").items()},
        support_network=_mapping("support_network"),
    )


def reviewed_inventory() -> tuple[ReviewedPart, ...]:
    """Every reviewed record the pipeline may select a part from.

    The three vendored sets were kept apart for provenance -- device records, the stock KiCad
    pairs the lowerers emit, and the validated stock/common identities -- but the selection-facing
    lookups read only the first. That made 25 portable records invisible, the two reviewed
    Schottky order codes (B340A, SS34: symbol, footprint and ratings all recorded) among them: a
    brief that demanded a Schottky was told the library carried no such part while the library
    carried two. Selection now reads all of them plus anything this machine researched for itself;
    rows that are not placeable (an identity-only package boundary) are filtered by the caller's
    own ``is_portable_candidate`` test where a real part is required.
    """
    return (
        *REVIEWED_PARTS,
        *_STANDARD_LIBRARY_PARTS,
        *_STOCK_COMMON_PARTS,
        *_researched_parts(),
    )


def reviewed_part(identity: str) -> ReviewedPart | None:
    """Return the exact reviewed record, never a prefix or family inference."""
    key = identity.strip().casefold()
    return next((part for part in reviewed_inventory() if part.identity == key), None)


# A small number of reviewed rows are selected by the manufacturer's bare order
# code while the inventory identity retains its vendor-qualified name.  These
# are explicit aliases, not normalization rules: physical_inventory_record()
# still requires the reviewed symbol/footprint pair before accepting either
# spelling, so a coincidental catalog code cannot bless another part.
_REVIEWED_ORDER_CODE_IDENTITIES: dict[str, str] = {
    "8734": "keystone-8734",
}


def reviewed_identity_for_order_code(identity: str) -> str:
    """Resolve one explicitly reviewed manufacturer order-code spelling."""
    key = identity.strip().casefold()
    return _REVIEWED_ORDER_CODE_IDENTITIES.get(key, key)


# These are the stock KiCad physical pairs emitted directly by the lowerers,
# plus the reviewed order codes the validated reference rows assert on those
# same stock pairs (the 2026-09-16 block at the end of the tuple, which also
# carries the two repository bundles those rows ship).  They are deliberately
# exact source-library identities, not text heuristics: a caller must supply
# the exact symbol and footprint below, and may not attach an incompatible
# explicit MPN.  The header shape is checked independently so that, for
# example, a 1x03 symbol cannot claim a 1x04 footprint.
_STANDARD_LIBRARY_PARTS: tuple[ReviewedPart, ...] = (
    ReviewedPart(
        identity="12401610e4-2a",
        family="usb-c-receptacle",
        package="Amphenol 12401610E4-2A USB-C receptacle",
        bundle="kicad-standard",
        symbol="Connector:USB_C_Receptacle",
        footprint="Connector_USB:USB_C_Receptacle_Amphenol_12401610E4-2A",
        physical_features=frozenset({"usb-c-receptacle"}),
        contacts=(
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A11",
            "A12",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B9",
            "B10",
            "B11",
            "B12",
            "S1",
        ),
        manufacturer_sources=(
            "https://cdn.amphenol-cs.com/media/wysiwyg/files/drawing/c12401610_c.pdf",
        ),
    ),
    ReviewedPart(
        identity="bs-07-a1bj001",
        family="coin-cell-holder",
        package="MYOUNG BS-07-A1BJ001 CR2032 holder",
        bundle="kicad-standard",
        symbol="Device:Battery_Cell",
        footprint="Battery:BatteryHolder_MYOUNG_BS-07-A1BJ001_CR2032",
        physical_features=frozenset({"coin-cell-holder"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C2979167.pdf",),
        lcsc="C2979167",
    ),
    ReviewedPart(
        identity="type-c-31-m-12",
        family="usb-c-receptacle",
        package="16-pin USB-C receptacle with shield retention tabs",
        bundle="usb-c-16p",
        symbol="usb-c-16p:TYPE-C-31-M-12",
        footprint="usb-c-16p:USB-C_SMD-TYPE-C-31-M-12_1",
        physical_features=frozenset({"usb-c-receptacle"}),
        contacts=(
            "1",
            "2",
            "3",
            "4",
            "A5",
            "A6",
            "A7",
            "A8",
            "B5",
            "B6",
            "B7",
            "B8",
            "A4B9",
            "B4A9",
            "A1B12",
            "B1A12",
        ),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C165948.pdf",),
        lcsc="C165948",
        operating_limits={"voltage_v": 20, "current_a": 5},
    ),
    # --- Reviewed order codes on stock land patterns (2026-09-16) -------------
    # Each record below is an order code the validated reference rows already
    # assert, together with the exact stock KiCad symbol and footprint those
    # rows emit.  The rows carry the manufacturer, package, LCSC code and
    # ratings in their own part_inventory facts and sourcing notes: the URL on
    # each record is that row's recorded source, the LCSC code is the one the
    # row's note pins (the read-only JLC/LCSC dump re-checks it and agrees on
    # the stock figure), and no rating is carried across from a sibling order
    # code.  This is the identity the registry was missing, not a relaxation:
    # a caller still must supply the exact pair named here.
    #
    # Three rows assert a catalog spelling rather than a clean order code
    # ("CD54-100M 10UH", "RVT1H220M0605 22UF 50V" and
    # "JBLH2101M050C120RLM 100UF 50V").  Those are registered exactly as the
    # row writes them -- a re-spelling would not resolve the group -- while the
    # package text names the orderable series and the bundle/catalog source.
    ReviewedPart(
        identity="37205000001",
        family="fuse",
        package="Littelfuse 372 series radial lead cartridge fuse, 8.5 x 8 mm body, 5.08 mm pitch",
        bundle="kicad-standard",
        symbol="Device:Fuse",
        footprint="Fuse:Fuse_Littelfuse_372_D8.50mm",
        physical_features=frozenset({"fuse", "slow-blow-fuse", "input-fuse"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588891356407713792",
        ),
        lcsc="C142835",
        operating_limits={
            "current_a": 0.5,
            "voltage_v": 250.0,
            "breaking_capacity_a": 35.0,
            "temperature_min_c": -40.0,
            "temperature_max_c": 85.0,
        },
    ),
    ReviewedPart(
        identity="ac0603fr-071kl",
        family="resistor-0603",
        package="YAGEO AC series 0603 (1608 metric) thick-film resistor",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric",
        physical_features=frozenset({"resistor-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8754852008425861120",
        ),
        lcsc="C116692",
        operating_limits={
            "resistance_ohm": 1000.0,
            "tolerance_fraction": 0.01,
            "power_w": 0.1,
            "voltage_v": 75.0,
            "temperature_min_c": -55.0,
            "temperature_max_c": 155.0,
        },
    ),
    ReviewedPart(
        identity="b340a",
        family="schottky-diode",
        package="MDD Microdiode B340A SMA (DO-214AC) Schottky rectifier",
        bundle="kicad-standard",
        symbol="Device:D_Schottky",
        footprint="Diode_SMD:D_SMA",
        physical_features=frozenset({"schottky-diode", "rectifier-diode"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8586175732742049792",
        ),
        lcsc="C64982",
        operating_limits={
            "forward_current_a": 3.0,
            "reverse_voltage_v": 40.0,
            "forward_voltage_v": 0.55,
            "temperature_min_c": -50.0,
            "temperature_max_c": 150.0,
        },
    ),
    ReviewedPart(
        identity="bzt52c10",
        family="zener-diode",
        package="MDD Microdiode BZT52C10 SOD-123 zener diode",
        bundle="kicad-standard",
        symbol="Device:D_Zener",
        footprint="Diode_SMD:D_SOD-123",
        physical_features=frozenset({"zener-diode"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8756679576573890560",
        ),
        lcsc="C173431",
        operating_limits={
            "zener_voltage_v": 10.0,
            "zener_voltage_min_v": 9.5,
            "zener_voltage_max_v": 10.5,
            "power_w": 0.5,
        },
    ),
    ReviewedPart(
        identity="cc0805kkx7r0bb105",
        family="capacitor-0805",
        package="YAGEO CC series 0805 (2012 metric) X7R MLCC",
        bundle="kicad-standard",
        symbol="Device:C",
        footprint="Capacitor_SMD:C_0805_2012Metric",
        physical_features=frozenset({"capacitor-0805"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8735625186459312128",
        ),
        lcsc="C5370002",
        operating_limits={
            "capacitance_uf": 1.0,
            "voltage_v": 100.0,
            "tolerance_fraction": 0.1,
            "dielectric": "X7R",
        },
    ),
    ReviewedPart(
        identity="cd54-100m 10uh",
        family="power-inductor",
        package=(
            "DMBJ CD54-100M unshielded SMD power inductor, 5.8 x 5.2 mm; the row "
            "asserts this catalog spelling of the bundle's own order code "
            "(bundle cd54-100m-inductor, manifest mpn 'CD54-100M 10UH')"
        ),
        bundle="cd54-100m-inductor",
        symbol="cd54-100m-inductor:CD54-100M",
        footprint="cd54-100m-inductor:IND-SMD_L5.8-W5.2",
        physical_features=frozenset({"power-inductor", "filter-inductor"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C2826662.pdf",),
        lcsc="C2826662",
        operating_limits={
            "inductance_uh": 10.0,
            "tolerance_fraction": 0.2,
            "current_a": 1.44,
            "dc_resistance_ohm": 0.1,
        },
    ),
    ReviewedPart(
        identity="ds1034-09funsi44",
        family="de-9-connector",
        package="CONNFLY DS1034-09FUNSi44 through-hole 9-position standard D-Sub socket",
        bundle="kicad-standard",
        symbol="Connector:DE9_Socket",
        footprint="Connector_Dsub:DSUB-9_Socket_Vertical_P2.77x2.84mm",
        physical_features=frozenset({"db9-connector", "d-sub-connector"}),
        contacts=tuple(str(number) for number in range(1, 10)),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8563423381518241792",
        ),
        lcsc="C77831",
        operating_limits={"temperature_min_c": -40.0, "temperature_max_c": 105.0},
    ),
    ReviewedPart(
        identity="frc0603f1002ts",
        family="resistor-0603",
        package="FOJAN FRC series 0603 (1608 metric) thick-film resistor",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric",
        physical_features=frozenset({"resistor-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/partdetail/FOJAN-FRC0603F1002TS/C2906982",
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8763490791224557568",
        ),
        lcsc="C2906982",
        operating_limits={
            "resistance_ohm": 10000.0,
            "tolerance_fraction": 0.01,
            "power_w": 0.1,
            "voltage_v": 75.0,
        },
    ),
    ReviewedPart(
        identity="frc0603f5901ts",
        family="resistor-0603",
        package="FOJAN FRC series 0603 (1608 metric) thick-film resistor",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric",
        physical_features=frozenset({"resistor-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8763495553592778752",
        ),
        lcsc="C2933242",
        operating_limits={
            "resistance_ohm": 5900.0,
            "tolerance_fraction": 0.01,
            "power_w": 0.1,
            "voltage_v": 75.0,
        },
    ),
    ReviewedPart(
        identity="fxl0630-150-m",
        family="power-inductor",
        package=(
            "cjiang FXL0630 molded SMD power inductor, 7.0 x 6.6 mm "
            "(bundle inductor-15uh-3a, footprint IND-SMD_L7.0-W6.6_FXL0630)"
        ),
        bundle="inductor-15uh-3a",
        symbol="inductor-15uh-3a:FXL0630-150-M",
        footprint="inductor-15uh-3a:IND-SMD_L7.0-W6.6_FXL0630",
        physical_features=frozenset({"power-inductor", "buck-inductor"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://lcsc.com/product-detail/Power-Inductors_15uH-20_C177243.html",
        ),
        lcsc="C177243",
        operating_limits={
            "inductance_uh": 15.0,
            "tolerance_fraction": 0.2,
            "current_a": 3.0,
            "saturation_current_a": 4.0,
            "dc_resistance_ohm": 0.115,
        },
    ),
    ReviewedPart(
        identity="grm32dr71e106ka12l",
        family="capacitor-1210",
        package="Murata GRM32 series 1210 (3225 metric) X7R MLCC",
        bundle="kicad-standard",
        symbol="Device:C",
        footprint="Capacitor_SMD:C_1210_3225Metric",
        physical_features=frozenset({"capacitor-1210"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588919302553030656",
        ),
        lcsc="C77100",
        operating_limits={
            "capacitance_uf": 10.0,
            "voltage_v": 25.0,
            "tolerance_fraction": 0.1,
            "dielectric": "X7R",
        },
    ),
    ReviewedPart(
        identity="jblh2101m050c120rlm 100uf 50v",
        family="electrolytic-capacitor",
        package=(
            "DMBJ JBLH series radial through-hole aluminium electrolytic, "
            "D6.3 x L12 mm, 2.50 mm pitch; the row asserts this catalog spelling"
        ),
        bundle="kicad-standard",
        symbol="Device:C_Polarized",
        footprint="Capacitor_THT:CP_Radial_D6.3mm_P2.50mm",
        physical_features=frozenset({"electrolytic-capacitor", "output-bulk-capacitor"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8602919826839531520",
        ),
        lcsc="C19270686",
        operating_limits={
            "capacitance_uf": 100.0,
            "voltage_v": 50.0,
            "tolerance_fraction": 0.2,
        },
    ),
    ReviewedPart(
        identity="mmbt3904",
        family="npn-transistor",
        package="SOT-23 NPN bipolar transistor (row manufacturer string: ST Semtech)",
        bundle="kicad-standard",
        symbol="Transistor_BJT:MMBT3904",
        footprint="Package_TO_SOT_SMD:SOT-23",
        physical_features=frozenset({"npn-transistor", "bjt"}),
        contacts=("1", "2", "3"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8586176249693941760",
        ),
        lcsc="C84104",
        operating_limits={
            "vceo_v": 40.0,
            "collector_current_a": 0.2,
            "hfe_min": 100.0,
            "power_w": 0.35,
            "transition_frequency_mhz": 300.0,
        },
    ),
    # The nrf52-beacon row asserts an engineering value only for its LFXO
    # (32.768 kHz, 9 pF load, 20 ppm) on the stock two-pad 3215 land pattern and
    # names no order code, so this record comes from the read-only JLC/LCSC
    # catalog row for the order code itself: NDK NX3215SA-32.768K-STD-MUA-9,
    # SMD3215-2P, 9 pF, +/-20 ppm, LCSC C519280.  NDK's "-MUA-8" (12.5 pF) and
    # "-MUA-14" (6 pF) are different load-capacitance order codes and are NOT
    # members: only the 9 pF part matches the row's declared load.
    ReviewedPart(
        identity="nx3215sa-32.768k-std-mua-9",
        family="crystal-3215",
        package="NDK NX3215SA 3.2 x 1.5 mm two-pad SMD crystal, 9 pF load",
        bundle="kicad-standard",
        symbol="Device:Crystal",
        footprint="Crystal:Crystal_SMD_3215-2Pin_3.2x1.5mm",
        physical_features=frozenset({"crystal", "crystal-3215"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588948980885827584",
        ),
        lcsc="C519280",
        operating_limits={
            "frequency_khz": 32.768,
            "load_capacitance_pf": 9.0,
            "tolerance_ppm": 20.0,
            "temperature_min_c": -40.0,
            "temperature_max_c": 85.0,
        },
    ),
    ReviewedPart(
        identity="rc0603fr-07100kl",
        family="resistor-0603",
        package="YAGEO RC series 0603 (1608 metric) thick-film resistor",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric",
        physical_features=frozenset({"resistor-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8755823835092664320",
        ),
        lcsc="C14675",
        operating_limits={
            "resistance_ohm": 100000.0,
            "tolerance_fraction": 0.01,
            "power_w": 0.1,
            "voltage_v": 75.0,
            "temperature_min_c": -55.0,
            "temperature_max_c": 155.0,
        },
    ),
    ReviewedPart(
        identity="rvt1h220m0605 22uf 50v",
        family="electrolytic-capacitor",
        package=(
            "DMBJ RVT series SMD aluminium electrolytic, D6.3 x L5.4 mm; the row "
            "asserts this catalog spelling"
        ),
        bundle="kicad-standard",
        symbol="Device:C_Polarized",
        footprint="Capacitor_SMD:CP_Elec_6.3x5.4",
        physical_features=frozenset({"electrolytic-capacitor", "output-ripple-capacitor"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8590164184962412544",
        ),
        lcsc="C970664",
        operating_limits={
            "capacitance_uf": 22.0,
            "voltage_v": 50.0,
            "tolerance_fraction": 0.2,
            "endurance_h_at_105c": 2800.0,
            "temperature_min_c": -55.0,
            "temperature_max_c": 105.0,
        },
    ),
    # The dual-rail-supply row's sourcing note is this record's evidence: CIN1 is
    # ROQANG RVT2A100M0607 (LCSC C72482, assembly stock 121902 at review), the
    # same order code and 6.3 x 7.7 mm land pattern as the DMBJ-sourced row
    # (LCSC C970659) whose stock fails the assembly floor.  The catalog row for
    # C72482 carries 100 V, 10 uF, +/-20 %, 2000 h at 105 C, -40 to +105 C.
    ReviewedPart(
        identity="rvt2a100m0607",
        family="electrolytic-capacitor",
        package="ROQANG RVT series SMD aluminium electrolytic, D6.3 x L7.7 mm (LCSC C72482)",
        bundle="kicad-standard",
        symbol="Device:C_Polarized",
        footprint="Capacitor_SMD:CP_Elec_6.3x7.7",
        physical_features=frozenset({"electrolytic-capacitor", "input-bulk-capacitor"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8756924718023237632",
        ),
        lcsc="C72482",
        operating_limits={
            "capacitance_uf": 10.0,
            "voltage_v": 100.0,
            "tolerance_fraction": 0.2,
            "endurance_h_at_105c": 2000.0,
            "temperature_min_c": -40.0,
            "temperature_max_c": 105.0,
        },
    ),
    ReviewedPart(
        identity="smdri127-220mt",
        family="power-inductor",
        package="SXN SMDRI127 shielded SMD power inductor, 12.3 x 12.3 mm",
        bundle="kicad-standard",
        symbol="Device:L",
        footprint="Inductor_SMD:L_12x12mm_H8mm",
        physical_features=frozenset({"power-inductor", "buck-inductor", "shielded-inductor"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588918807184756736",
        ),
        lcsc="C9905",
        operating_limits={
            "inductance_uh": 22.0,
            "tolerance_fraction": 0.2,
            "current_a": 3.6,
            "saturation_current_a": 7.0,
            "dc_resistance_ohm": 0.043,
        },
    ),
    ReviewedPart(
        identity="spz1cm221e08o00raxxx",
        family="electrolytic-capacitor",
        package=(
            "AISHI SPZ series radial through-hole polymer aluminium electrolytic, "
            "D6.3 x L8 mm, 2.5 mm pitch"
        ),
        bundle="kicad-standard",
        symbol="Device:C_Polarized",
        footprint="Capacitor_THT:CP_Radial_D6.3mm_P2.50mm",
        physical_features=frozenset(
            {"electrolytic-capacitor", "polymer-capacitor", "output-capacitor"}
        ),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588887533229125632",
        ),
        lcsc="C122241",
        operating_limits={
            "capacitance_uf": 220.0,
            "voltage_v": 16.0,
            "tolerance_fraction": 0.2,
            "esr_ohm_at_100khz": 0.02,
            "ripple_current_a_at_100khz": 2.7,
            "temperature_min_c": -55.0,
            "temperature_max_c": 105.0,
        },
    ),
    ReviewedPart(
        identity="ss34",
        family="schottky-diode",
        package="MDD Microdiode SS34 SMA (DO-214AC) Schottky rectifier",
        bundle="kicad-standard",
        symbol="Device:D_Schottky",
        footprint="Diode_SMD:D_SMA",
        physical_features=frozenset({"schottky-diode", "rectifier-diode"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8586172667489079296",
        ),
        lcsc="C8678",
        operating_limits={
            "forward_current_a": 3.0,
            "reverse_voltage_v": 40.0,
            "forward_voltage_v": 0.55,
            "temperature_min_c": -55.0,
            "temperature_max_c": 125.0,
        },
    ),
    # The nrf52-beacon user_button group names the E-Switch TL3342 series and
    # LCSC C2886894 (its datasheet link) on this stock pair; the read-only
    # JLC/LCSC catalog row for C2886894 is TL3342F260QG, 5.2 x 5.2 mm four-pad
    # SPST-NO tactile switch.  "TL3342" alone is the series designation, so
    # _DEVICE_MEMBERS routes that request here instead of the stock
    # kicad-tl3342-button land-pattern id, which is not orderable hardware.
    ReviewedPart(
        identity="tl3342f260qg",
        family="momentary-button",
        package="E-Switch TL3342 series four-pad SPST-NO SMD tactile switch, 5.2 x 5.2 mm",
        bundle="kicad-standard",
        symbol="Switch:SW_Push",
        footprint="Button_Switch_SMD:SW_SPST_TL3342",
        physical_features=frozenset({"momentary-button", "spst-switch", "tactile-switch"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://www.lcsc.com/datasheet/C2886894.pdf",),
        lcsc="C2886894",
        operating_limits={
            "voltage_v": 12.0,
            "current_a": 0.05,
            "actuation_force_n": 2.6,
            "operating_life_cycles": 100000.0,
            "temperature_min_c": -20.0,
            "temperature_max_c": 70.0,
        },
    ),
    ReviewedPart(
        identity="wsl2512r1000fea",
        family="resistor-2512",
        package="Vishay WSL series 2512 (6332 metric) current-sense resistor",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_2512_6332Metric",
        physical_features=frozenset({"resistor-2512", "current-sense-resistor"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588893225277276160",
        ),
        lcsc="C844904",
        operating_limits={
            "resistance_ohm": 0.1,
            "tolerance_fraction": 0.01,
            "power_w": 1.0,
            "temperature_coefficient_ppm_per_c": 75.0,
        },
    ),
)

# These unbranded stock pairs are accepted only when no MPN is asserted.
# They evidence a finite, exact KiCad land-pattern/symbol contract; they are
# not manufacturer substitutions or a basis for inferring a part family.
_STOCK_COMMON_PARTS: tuple[ReviewedPart, ...] = (
    ReviewedPart(
        identity="kicad-resistor-0603",
        family="resistor-0603",
        package="0603 (1608 metric) two-pad resistor",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric",
        physical_features=frozenset({"resistor-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://gitlab.com/kicad/libraries/kicad-footprints",),
    ),
    ReviewedPart(
        identity="kicad-resistor-1210",
        family="resistor-1210",
        package="1210 (3225 metric) two-pad resistor",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_1210_3225Metric",
        physical_features=frozenset({"resistor-1210"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://gitlab.com/kicad/libraries/kicad-footprints",),
    ),
    ReviewedPart(
        identity="kicad-capacitor-0603",
        family="capacitor-0603",
        package="0603 (1608 metric) two-pad capacitor",
        bundle="kicad-standard",
        symbol="Device:C",
        footprint="Capacitor_SMD:C_0603_1608Metric",
        physical_features=frozenset({"capacitor-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://gitlab.com/kicad/libraries/kicad-footprints",),
    ),
    ReviewedPart(
        identity="kicad-led-0603",
        family="led-0603",
        package="0603 (1608 metric) two-pad LED",
        bundle="kicad-standard",
        symbol="Device:LED",
        footprint="LED_SMD:LED_0603_1608Metric",
        physical_features=frozenset({"led-0603"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://gitlab.com/kicad/libraries/kicad-symbols",
            "https://gitlab.com/kicad/libraries/kicad-footprints",
        ),
    ),
    ReviewedPart(
        identity="kicad-testpoint-pad-d1mm",
        family="test-point",
        package="1.00 mm single-pad test point",
        bundle="kicad-standard",
        symbol="Connector:TestPoint",
        footprint="TestPoint:TestPoint_Pad_D1.0mm",
        physical_features=frozenset({"test-point"}),
        contacts=("1",),
        manufacturer_sources=(
            "https://gitlab.com/kicad/libraries/kicad-symbols",
            "https://gitlab.com/kicad/libraries/kicad-footprints",
        ),
    ),
    ReviewedPart(
        identity="kicad-tl3342-button",
        family="momentary-button",
        package="TL3342-pattern four-pad SPST momentary button",
        bundle="kicad-standard",
        symbol="Switch:SW_Push",
        footprint="Button_Switch_SMD:SW_SPST_TL3342",
        physical_features=frozenset({"momentary-button", "spst-switch"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://gitlab.com/kicad/libraries/kicad-symbols",
            "https://gitlab.com/kicad/libraries/kicad-footprints",
        ),
    ),
    # 0402 passives and the installed SMD crystal land patterns.  These are
    # stock land-pattern contracts only: no manufacturer identity is asserted,
    # exactly like the 0603 entries above.  The 2.0 x 1.6 mm body ships as a
    # four-pad land pattern in the installed KiCad library (case pads 3 and 4
    # are not signal contacts) and that is the id registered below; the 2-pin
    # 2.0 x 1.6 mm id does not exist there and is deliberately NOT registered,
    # because it would name a library id that cannot be resolved.  The 2-pin
    # 3.2 x 1.5 mm body does exist and the lfxo NX3215SA uses it.
    ReviewedPart(
        identity="kicad-resistor-0402",
        family="resistor-0402",
        package="0402 (1005 metric) two-pad resistor",
        bundle="kicad-standard",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0402_1005Metric",
        physical_features=frozenset({"resistor-0402"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://gitlab.com/kicad/libraries/kicad-footprints",),
    ),
    ReviewedPart(
        identity="kicad-capacitor-0402",
        family="capacitor-0402",
        package="0402 (1005 metric) two-pad capacitor",
        bundle="kicad-standard",
        symbol="Device:C",
        footprint="Capacitor_SMD:C_0402_1005Metric",
        physical_features=frozenset({"capacitor-0402"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://gitlab.com/kicad/libraries/kicad-footprints",),
    ),
    ReviewedPart(
        identity="kicad-inductor-0402",
        family="inductor-0402",
        package="0402 (1005 metric) two-pad inductor",
        bundle="kicad-standard",
        symbol="Device:L",
        footprint="Inductor_SMD:L_0402_1005Metric",
        physical_features=frozenset({"inductor-0402"}),
        contacts=("1", "2"),
        manufacturer_sources=("https://gitlab.com/kicad/libraries/kicad-footprints",),
    ),
    ReviewedPart(
        identity="kicad-crystal-3215-2pin",
        family="crystal-3215",
        package="3.2 x 1.5 mm two-pin SMD crystal land pattern",
        bundle="kicad-standard",
        symbol="Device:Crystal",
        footprint="Crystal:Crystal_SMD_3215-2Pin_3.2x1.5mm",
        physical_features=frozenset({"crystal", "crystal-3215"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://gitlab.com/kicad/libraries/kicad-symbols",
            "https://gitlab.com/kicad/libraries/kicad-footprints",
        ),
    ),
    ReviewedPart(
        identity="kicad-crystal-2016-4pin",
        family="crystal-2016",
        package=(
            "2.0 x 1.6 mm four-pad SMD crystal land pattern; pads 3 and 4 are "
            "the case pads, not signal contacts"
        ),
        bundle="kicad-standard",
        symbol="Device:Crystal",
        footprint="Crystal:Crystal_SMD_2016-4Pin_2.0x1.6mm",
        physical_features=frozenset({"crystal", "crystal-2016"}),
        contacts=("1", "2"),
        manufacturer_sources=(
            "https://gitlab.com/kicad/libraries/kicad-symbols",
            "https://gitlab.com/kicad/libraries/kicad-footprints",
        ),
        support_network={
            "case_pads": ("3", "4"),
            "case_pads_are_no_connect": True,
        },
    ),
    # A mounting hole is a bare-board feature, never orderable hardware: it is
    # registered only on the no-MPN path (an asserted MPN cannot reach this
    # tuple) and carries the reviewed mounting-hole feature the geometry gate
    # reads.  It asserts no manufacturer identity and no electrical contact.
    ReviewedPart(
        identity="kicad-mounting-hole-3.2mm-m3",
        family="mounting-hole",
        package="3.2 mm non-plated M3 mounting hole",
        bundle="kicad-standard",
        symbol="Mechanical:MountingHole",
        footprint="MountingHole:MountingHole_3.2mm_M3",
        physical_features=frozenset({"mounting-hole"}),
        contacts=(),
        manufacturer_sources=("https://gitlab.com/kicad/libraries/kicad-footprints",),
    ),
)

_STOCK_TERMINAL_SYMBOL_RE = re.compile(r"Connector:Screw_Terminal_01x(?P<pins>\d{2})")
# The installed KiCad library zero-pads the pole count in the footprint name
# ("..._1x03_P5.00mm") while the symbol uses two digits as well, so the two
# counts are captured independently and compared numerically rather than by a
# textual backreference.
_STOCK_TERMINAL_FOOTPRINT_RE = re.compile(
    r"TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-"
    r"(?P<positions>\d+)_1x(?P<pins>\d+)_P5\.00mm_Horizontal"
)

_STOCK_HEADER_SYMBOL_RE = re.compile(
    r"Connector_Generic:Conn_(?P<rows>01|02)x(?P<pins>\d{2})"
    r"(?:_(?:Odd_Even|Counter_Clockwise))?"
)
_STOCK_HEADER_FOOTPRINT_RE = re.compile(
    r"Connector_Pin(?P<kind>Header|Socket)_2\.54mm:"
    r"Pin(?P=kind)_(?P<rows>1|2)x(?P<pins>\d{2})_P2\.54mm_Vertical"
)

# The Arduino Uno shield template emits exactly these pin-socket pairs for its
# four stacking connectors (form_factors._ARDUINO_UNO_SHIELD.fixed_connectors,
# realized by lowering._connector's female branch: digital_high 1x10,
# digital_low 1x08, power 1x08, analog 1x06).  A pin socket in that template IS
# the stacking interface the shield brief asks for, so these three pairs carry
# the reviewed stacking-header feature.  A generic 2.54 mm socket does not.
_ARDUINO_SHIELD_STACKING_PAIRS = frozenset(
    {
        (
            "Connector_Generic:Conn_01x10",
            "Connector_PinSocket_2.54mm:PinSocket_1x10_P2.54mm_Vertical",
        ),
        (
            "Connector_Generic:Conn_01x08",
            "Connector_PinSocket_2.54mm:PinSocket_1x08_P2.54mm_Vertical",
        ),
        (
            "Connector_Generic:Conn_01x06",
            "Connector_PinSocket_2.54mm:PinSocket_1x06_P2.54mm_Vertical",
        ),
    }
)


# The physical features each stock pattern record below can carry. They live here, not
# inline in the builder, because reviewed_feature_vocabulary() (and through it the
# realizable-class gate) must know every feature a reviewed record can carry without
# re-deriving these regexes.
_TERMINAL_PATTERN_FEATURES = frozenset({"screw-terminal", "terminal-block", "power-connector"})
_HEADER_PATTERN_FEATURES = frozenset({"pin-header", "header"})
_STACKING_HEADER_FEATURE = "stacking-header"
_LED0805_PATTERN_FEATURES = frozenset({"led-0805"})


def _stock_library_physical_record(symbol: str, footprint: str) -> ReviewedPart | None:
    """Return an evidence-backed stock KiCad record for a canonical pair."""
    common = next(
        (
            candidate
            for candidate in _STOCK_COMMON_PARTS
            if candidate.symbol == symbol and candidate.footprint == footprint
        ),
        None,
    )
    if common is not None:
        return common

    terminal_symbol = _STOCK_TERMINAL_SYMBOL_RE.fullmatch(symbol)
    terminal_footprint = _STOCK_TERMINAL_FOOTPRINT_RE.fullmatch(footprint)
    if terminal_symbol and terminal_footprint:
        contacts_count = int(terminal_symbol["pins"])
        if contacts_count != int(terminal_footprint["pins"]) or not 2 <= contacts_count <= 12:
            return None
        return ReviewedPart(
            identity=f"kicad-phoenix-mkds-1,5-{contacts_count:02d}",
            family="screw-terminal",
            package=f"Phoenix MKDS-1,5 1x{contacts_count:02d}, 5.00 mm-pitch terminal",
            bundle="kicad-standard",
            symbol=symbol,
            footprint=footprint,
            physical_features=_TERMINAL_PATTERN_FEATURES,
            contacts=tuple(str(number) for number in range(1, contacts_count + 1)),
            manufacturer_sources=(
                "https://gitlab.com/kicad/libraries/kicad-symbols",
                "https://gitlab.com/kicad/libraries/kicad-footprints",
            ),
        )

    header_symbol = _STOCK_HEADER_SYMBOL_RE.fullmatch(symbol)
    header_footprint = _STOCK_HEADER_FOOTPRINT_RE.fullmatch(footprint)
    if header_symbol and header_footprint:
        rows = int(header_symbol["rows"])
        pins_per_row = int(header_symbol["pins"])
        contact_count = rows * pins_per_row
        if (
            rows != int(header_footprint["rows"])
            or pins_per_row != int(header_footprint["pins"])
            or not 1 <= contact_count <= 40
        ):
            return None
        contacts = tuple(str(number) for number in range(1, contact_count + 1))
        features = set(_HEADER_PATTERN_FEATURES)
        if (symbol, footprint) in _ARDUINO_SHIELD_STACKING_PAIRS:
            features.add(_STACKING_HEADER_FEATURE)
        return ReviewedPart(
            identity=f"kicad-{header_footprint['kind'].casefold()}-{header_symbol['rows']}x{header_symbol['pins']}",
            family="pin-header",
            package=f"2.54 mm {header_footprint['kind'].casefold()} {header_symbol['rows']}x{header_symbol['pins']}",
            bundle="kicad-standard",
            symbol=symbol,
            footprint=footprint,
            physical_features=frozenset(features),
            contacts=contacts,
            manufacturer_sources=(
                "https://gitlab.com/kicad/libraries/kicad-symbols",
                "https://gitlab.com/kicad/libraries/kicad-footprints",
            ),
        )
    if symbol == "Device:LED" and footprint == "LED_SMD:LED_0805_2012Metric":
        return ReviewedPart(
            identity="kicad-led-0805",
            family="led-0805",
            package="0805 (2012 metric) LED",
            bundle="kicad-standard",
            symbol=symbol,
            footprint=footprint,
            physical_features=_LED0805_PATTERN_FEATURES,
            contacts=("1", "2"),
            manufacturer_sources=(
                "https://gitlab.com/kicad/libraries/kicad-symbols",
                "https://gitlab.com/kicad/libraries/kicad-footprints",
            ),
        )
    return None


# Obligation classes the model names that the reviewed feature vocabulary spells differently. Only
# pairs verified to denote the same physical class belong here: `usb-connector` is a USB-A part, so
# it is deliberately NOT an alias for a USB-C demand; `opto-isolator` is not a `digital-isolator`.
#
# Both consumers of the reviewed vocabulary resolve a demanded class through this one map — the BOM
# work-unit obligation check and the §9.42 physical-realization commit gate — so a class can never be
# aliased in one gate and unknown in the other (three-position-selector-switch passed neither until
# it was aliased here).
_DEMANDED_CLASS_ALIASES: dict[str, frozenset[str]] = {
    # The model's own shorthands for a class it names elsewhere in full.
    "fpc": frozenset({"fpc-connector"}),
    "header": frozenset({"pin-header", "pin-socket"}),
    "button": frozenset({"momentary-button"}),
    "pushbutton": frozenset({"momentary-button"}),
    "selector": frozenset({"three-position-selector", "sp3t-selector"}),
    "usb-c-connector": frozenset({"usb-c-receptacle"}),
    "usb-a-connector": frozenset({"usb-a-receptacle", "usb-connector"}),
    "usb_c_receptacle": frozenset({"usb-c-receptacle"}),
    # The corpus names the DC input connector both ways; both denote the one
    # physical class the reviewed DC005 carries.  `dc-power-jack` is the LCSC
    # category spelling of the same part.
    "barrel-jack": frozenset({"barrel-jack-connector"}),
    "dc-barrel-jack": frozenset({"barrel-jack-connector"}),
    "dc-power-jack": frozenset({"barrel-jack-connector"}),
    "fpc-ffc-connector": frozenset({"fpc-connector", "ffc-connector"}),
    "voltage-regulator-ic": frozenset({"voltage-regulator"}),
    "momentary-pushbutton": frozenset({"momentary-button"}),
    "status-led": frozenset({"indicator-led", "led-0603", "led-0805"}),
    "led": frozenset({"indicator-led", "led-0603", "led-0805"}),
    "power-led": frozenset({"indicator-led", "led-0603", "led-0805"}),
    "power-screw-terminal": frozenset({"screw-terminal", "terminal-block", "screw-clamp-terminal"}),
    "binding-post-terminal": frozenset({"binding-post", "screw-clamp-terminal"}),
    "buck-converter-ic": frozenset({"buck-converter", "buck-regulator"}),
    "thermocouple-input": frozenset({"thermocouple-converter"}),
    "high-side-load-switch": frozenset({"highside-switch"}),
    "opto-isolator": frozenset({"optocoupler"}),
    "lora-radio-module": frozenset({"lora-module", "spi-radio-module", "sx1276-module"}),
    "logic-input-header": frozenset({"header", "pin-header"}),
    "three-position-selector-switch": frozenset({"three-position-selector", "sp3t-selector"}),
    # The 2026-09-24 coverage audit (every `physical` obligation class across the
    # 2 826 saved states, joined with `has_reviewed_coverage`) found 190 demanded
    # classes with no carrier.  These are the ones that are a plain second spelling
    # of a class the library already implements, and that `reviewed_class_variants`
    # does NOT already repair (a class whose tokens are a superset of a reviewed
    # feature is told to use the reviewed name at the intent stage; an alias here
    # would suppress that repair and silently accept a looser demand).  Each target
    # below is verified to have at least one emittable record — an alias to a
    # feature with no symbol/footprint pair would turn the class's real-part
    # fallback into a permanent refusal, which is worse than the gap it closes.
    # The rest of the audit's list (jst-connector, and the device/board classes with
    # no reviewed carrier at all) is deliberately NOT aliased: no reviewed record
    # implements them, so the fallback is the honest route.
    "push-button": frozenset({"momentary-button"}),
    "op-amp": frozenset({"operational-amplifier"}),
    "gpio-expander": frozenset({"io-expander", "i2c-gpio-expander"}),
    "gpio-expander-ic": frozenset({"io-expander", "i2c-gpio-expander"}),
    "dc-dc-converter": frozenset(
        {"buck-converter", "isolated-dc-dc-converter", "dual-output-dc-dc-converter"}
    ),
    "qwiic-receptacle": frozenset({"qwiic-connector"}),
    "uln2003-driver": frozenset({"darlington-array"}),
    "rs-485-transceiver": frozenset({"rs485-transceiver"}),
    "current-limiting-switch": frozenset({"current-limited-power-switch"}),
    "current-limiter": frozenset({"current-limited-power-switch"}),
    "load-switch": frozenset({"highside-switch"}),
    "selector-switch": frozenset({"three-position-selector", "sp3t-selector"}),
    "audio-jack-3p5mm": frozenset({"audio-jack-3-5mm"}),
    "temperature-humidity-pressure-sensor": frozenset({"environmental-sensor", "i2c-sensor"}),
}


def canonical_physical_features(feature: str) -> frozenset[str]:
    """The reviewed feature names a demanded obligation class may be realized by.

    The demand is the caller's own wording (a brief says "a three-position selector switch"); the
    reviewed vocabulary spells the class its own way. A class with no verified alias is itself.
    """
    key = str(feature or "").strip().casefold()
    return _DEMANDED_CLASS_ALIASES.get(key, frozenset({key}))


# A topology lowerer can prove only the physical class its independently
# regenerated graph implements.  This vocabulary is shared by the bounded BOM
# work-unit gate and canonical §9.42 commit validation; it is never inferred
# from resistor count, reference prefix, or private payload metadata.
_TRUSTED_LOWERER_PHYSICAL_WITNESSES = frozenset(
    {
        ("r2r-ladder@1", "resistorladder"),
        ("r2r-ladder@1", "resistornetwork"),
        ("screw-terminal@1", "thermocoupleinput"),
    }
)


def lowerer_witnesses_physical_class(lowerer_id: str, component_class: str) -> bool:
    """Whether a registered lowerer topology proves this exact physical class."""
    return (
        str(lowerer_id).strip().casefold(),
        re.sub(r"[^a-z0-9]+", "", str(component_class).casefold()),
    ) in _TRUSTED_LOWERER_PHYSICAL_WITNESSES


# Every physical feature a reviewed record can carry. Both consumers of the reviewed
# vocabulary — the BOM work-unit obligation check (_group_has_physical_feature) and the
# §9.42 physical-realization gate — match a demand against a record's physical_features,
# so this union is exactly the set of demands some reviewed part can satisfy.
#
# Researched records join the union (a class this pipeline answered for itself is covered from
# then on), and the result is recomputed when the record file changes rather than fixed at import:
# a class researched mid-run must be visible to the next check in the same process.
_VOCABULARY_CACHE: tuple[tuple[ReviewedPart, ...], frozenset[str]] | None = None


def reviewed_feature_vocabulary() -> frozenset[str]:
    """Every physical feature a reviewed record carries, researched records included."""
    global _VOCABULARY_CACHE
    researched = _researched_parts()
    if _VOCABULARY_CACHE is not None and _VOCABULARY_CACHE[0] is researched:
        return _VOCABULARY_CACHE[1]
    vocabulary = frozenset().union(
        _TERMINAL_PATTERN_FEATURES,
        _HEADER_PATTERN_FEATURES,
        _LED0805_PATTERN_FEATURES,
        {_STACKING_HEADER_FEATURE},
        *(
            part.physical_features
            for part in (
                *REVIEWED_PARTS,
                *_STANDARD_LIBRARY_PARTS,
                *_STOCK_COMMON_PARTS,
                *researched,
            )
        ),
    )
    _VOCABULARY_CACHE = (researched, vocabulary)
    return vocabulary


def realizable_physical_features(component_class: str) -> frozenset[str]:
    """The reviewed features that could satisfy a demanded obligation class.

    An empty result means NO reviewed part can implement the demanded class, so the
    demand is unsatisfiable wherever it is checked (`physical-obligation-unfulfilled`
    at BOM, `E_PHYSICAL_REALIZATION` at commit) and must be re-worded by the stage that
    wrote it, not discovered later as an unrepairable work-unit defect.
    """
    return canonical_physical_features(component_class) & reviewed_feature_vocabulary()


def has_reviewed_coverage(component_class: str) -> bool:
    """Whether the reviewed library can answer for a demanded obligation class.

    False means no reviewed part carries the class: the demand is new ground, and every
    check that would otherwise demand reviewed evidence must fall back to
    :func:`resolved_part_evidence` instead of refusing it.
    """
    return bool(realizable_physical_features(component_class))


def _class_plural_singular(token: str) -> str:
    """``connectors`` -> ``connector``, ``terminals`` -> ``terminal``; ``bus`` stays ``bus``."""
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def class_key(value: str) -> str:
    """Comparable key for a part class or a phrase that names one.

    Folds case, separators and a trailing plural "s" per token, so the model's natural
    spelling of a class compares equal to the class itself: "JST-XH connectors" and
    "screw terminals" key to "jst-xh-connector" and "screw-terminal".

    Deliberately does NOT fall back to token overlap or substring matching. A quantity
    row's subject is often a *property* of one part rather than a count of that part
    ("pins on the 0.1 inch header", "fpc/ffc connector contacts"): those share a token
    with the class but mean eight pins on one header, so a fuzzy rule would demand eight
    headers. A phrase whose key differs from every class key is reported to the writer
    instead (:func:`quantity_subject_binds` consumers), never guessed at.
    """
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().casefold()).strip("-")
    return "-".join(_class_plural_singular(token) for token in text.split("-") if token)


#: Words a count's subject uses to describe a property OF something ("pins on the header",
#: "digits of the display") instead of naming a part class. A subject carrying one is the
#: writer's own property count: it never resolves to a class, and intent does not police it.
PROPERTY_PREPOSITIONS = frozenset({"of", "on", "in", "per", "for", "from", "within", "across"})


def _describes_a_property(subject: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", str(subject).casefold()))
    return bool(tokens & PROPERTY_PREPOSITIONS)


def quantity_subject_binds(subject: str, component_class: str) -> bool:
    """Whether a ``quantity`` row's subject names ``component_class``.

    Both consumers of a count (the BOM work-unit obligation check and the §9.42
    physical-realization gate) ask this one question, so a count can never bind in one
    gate and dangle in the other.
    """
    return quantity_class_for(subject, [component_class]) == component_class


def _alias_targets(value: str) -> frozenset[str]:
    """The reviewed classes a demanded-class spelling resolves to ("status led" -> led)."""
    key = class_key(value)
    targets: set[str] = set()
    for alias in (key, key.split("-")[-1]):
        for target in _DEMANDED_CLASS_ALIASES.get(alias, ()):
            targets.add(class_key(target))
    return frozenset(targets)


def quantity_class_for(subject: str, classes) -> str | None:
    """The ONE class in ``classes`` a quantity row's subject counts, else ``None``.

    Resolution order, most specific first:

    1. the same key -- case, separators and a trailing plural are folded ("screw terminals",
       "JST-XH connectors");
    2. the same reviewed class through the demanded-class alias map ("status led" and "led"
       both denote the reviewed LED classes).

    A subject that describes a property OF something ("pins on the 0.1 inch header") is never
    resolved through the alias map: its head noun names the container the property belongs to,
    not a class to count, so eight pins stay eight pins on one header. An ambiguous subject
    (two classes it could count) or one naming no class resolves to ``None``: the count binds
    to nothing, and the stage that wrote it is asked to name the class
    (``intent_quantity_subject_unbound``) rather than the pipeline guessing which class a
    count was for.
    """
    keys = class_key(subject)
    if not keys:
        return None
    exact = [value for value in classes if class_key(value) == keys]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    if _describes_a_property(subject):
        return None
    forms = {keys, *_alias_targets(subject)}
    matches = [
        value
        for value in classes
        if class_key(value) in forms or (_alias_targets(value) & forms)
    ]
    return matches[0] if len(matches) == 1 else None


#: How a reviewed supply limit domain reads in a message. A record names its supply domain
#: consistently, not identically: a logic part's `vin`/`input_voltage`, a motor driver's `vm`
#: (published as `motor_supply_*_v`).
_SUPPLY_DOMAIN_LABELS = {
    "motor_supply": "motor supply",
    "input_voltage": "input",
    "vin": "input",
    "vdd": "logic supply",
    "output_voltage": "output",
}


#: Which reviewed port carries each published limit domain. A record names its limit domain
#: and its supply port independently ("motor_supply_*_v" on "vm"; "supply_*_v" on "vdd"), so
#: the check maps the domain onto the port the record actually declares instead of trusting
#: either spelling alone. Ports the map does not know stay unrated: silence, never a guess.
_DOMAIN_PORT_CANDIDATES = {
    "motor_supply": ("vm", "motor_supply", "vs", "vmm"),
    "input_voltage": ("vin", "input", "input_positive"),
    "vin": ("vin", "input", "input_positive"),
    "input": ("vin", "input", "input_positive"),
    "supply": ("vdd", "vcc", "vin", "input", "vm", "vs"),
    "io_supply": ("vddio", "iovdd", "vdd", "vcc"),
    "output_voltage": ("vout", "output", "output_positive"),
}


def _record_field(record, name: str) -> dict:
    """A reviewed record's field, whether the caller holds the ``ReviewedPart`` or its ``vars()``."""
    value = record.get(name) if isinstance(record, dict) else getattr(record, name, None)
    return value if isinstance(value, dict) else {}


def reviewed_supply_voltage_limits(
    record,
) -> list[tuple[str, float | None, float | None]]:
    """Every supply domain a reviewed record publishes, as ``(label, min_v, max_v)``.

    Read straight from ``operating_limits`` so a record's own spelling reaches the checks:
    the DRV8833's ``motor_supply_min_v``/``motor_supply_max_v`` pair is exactly the fact that
    makes 18 V on its VM pin a fault, and the input-range check used to skip it because it
    only knew ``vin``/``input`` spellings. Sorted most restrictive first.
    """
    limits = _record_field(record, "operating_limits")
    rows: list[tuple[str, float | None, float | None]] = []
    for key, value in limits.items():
        if not str(key).endswith("_max_v"):
            continue
        try:
            maximum = float(value)
        except (TypeError, ValueError):
            continue
        domain = str(key)[: -len("_max_v")]
        try:
            minimum = float(limits[f"{domain}_min_v"])
        except (KeyError, TypeError, ValueError):
            minimum = None
        rows.append((_SUPPLY_DOMAIN_LABELS.get(domain, domain.replace("_", " ")), minimum, maximum))
    return sorted(rows, key=lambda row: (row[2], row[0]))


def reviewed_supply_port_limits(
    record,
) -> list[tuple[str, str, float | None, float | None]]:
    """``(port_key, label, min_v, max_v)`` for each rated supply domain the record also declares
    a port for. The record's ports decide which domain a rail is compared against, so a motor
    supply is checked against its motor rating rather than a logic rating."""
    limits = _record_field(record, "operating_limits")
    ports = _record_field(record, "port_pins")
    declared = {str(key).casefold(): str(key) for key in ports}
    rows: list[tuple[str, str, float | None, float | None]] = []
    for key, value in limits.items():
        if not str(key).endswith("_max_v"):
            continue
        domain = str(key)[: -len("_max_v")]
        try:
            maximum = float(value)
        except (TypeError, ValueError):
            continue
        port_key = next(
            (
                declared[candidate]
                for candidate in _DOMAIN_PORT_CANDIDATES.get(domain, ())
                if candidate in declared
            ),
            None,
        )
        if port_key is None:
            continue
        try:
            minimum = float(limits[f"{domain}_min_v"])
        except (KeyError, TypeError, ValueError):
            minimum = None
        label = _SUPPLY_DOMAIN_LABELS.get(domain, domain.replace("_", " "))
        rows.append((port_key, label, minimum, maximum))
    return sorted(rows, key=lambda row: (row[3], row[0]))


@lru_cache(maxsize=1)
def reviewed_class_heads() -> frozenset[str]:
    """The head nouns of the reviewed class vocabulary ("connector", "terminal", "led").

    A quantity subject ending in one of these reads as a part class ("JST-XH connectors",
    "3.5 mm audio jacks"); one that does not ("relay channels", "temperature settings") is a
    property or a design fact. The intent detector uses this to tell a lost part class from
    a count it should stay out of.
    """
    heads = set()
    for feature in reviewed_feature_vocabulary():
        key = class_key(feature)
        if key:
            heads.add(key.split("-")[-1])
    return frozenset(heads)


def resolved_part_evidence(*, mpn: str | None, symbol: str | None, footprint: str | None) -> bool:
    """Whether a BOM group is a real part that resolves, with no reviewed record.

    The evidence a demanded class the library has never covered is allowed to accept: an
    exact orderable identity (a named MPN, never a family label or a bare value) plus a
    symbol whose pin inventory resolves and a footprint to draw it with. A new part
    category is then built from a real part — the same standard the pipeline already
    applies to a part's declared wiring — instead of being refused.
    """
    if (
        not str(mpn or "").strip()
        or not str(symbol or "").strip()
        or not str(footprint or "").strip()
    ):
        return False
    # Local import: the symbol pinout layer sits above this one.
    from kicraft.design.synthesis.symbol_pinout import SymbolNotFoundError, lookup_pins

    try:
        pins = lookup_pins(str(symbol))
    except (SymbolNotFoundError, ValueError, OSError):
        return False
    return bool(pins.get("pins"))


def _class_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).strip().casefold()))


# Tokens that name a fact about the board or its wiring rather than a physical class: a bus
# ("i2c-interface"), a board format ("arduino-uno-format-board"), a package style, or a
# printed-copper feature ("thermal-via-copper-pour"). A demand carrying one belongs in
# constraints, a `fabrication` row, or a `negative` row instead.
_NOT_A_PART_CLASS_TOKENS = frozenset(
    {
        "interface",
        "bus",
        "protocol",
        "format",
        "outline",
        "layout",
        "shape",
        "package",
        "footprint",
        "pour",
        "plane",
        "via",
        "vias",
        "net",
        "netlist",
    }
)


def class_is_not_a_part(component_class: str) -> tuple[str, ...]:
    """The tokens that mark a demanded class as a board/wiring fact, not a part class."""
    return tuple(sorted(_class_tokens(component_class) & _NOT_A_PART_CLASS_TOKENS))


# Tokens naming the *source* of power rather than a part the board places: a pack, a cell, a bare
# battery. The board carries the mate -- a holder, a connector, a terminal -- and the source stays
# off-board, so no placed part can ever implement the demand. The real-part fallback cannot rescue
# it either: the group that honestly implements the input is a generic header or terminal, and a
# generic lowerer group carries no MPN to prove. Live run KC-5CNKJ3 asked a 2S Li-ion pack's BOM
# unit for `battery-pack` and exhausted all four attempts on it, while the corpus's battery-input
# designs that pass carry no such obligation at all -- the connector requirement alone.
_OFF_BOARD_SOURCE_TOKENS = frozenset(
    {"battery", "batteries", "cell", "cells", "pack", "packs", "accumulator"}
)

#: Tokens that name the board-side *mate* of a source rather than the source itself.
_SOURCE_MATE_TOKENS = frozenset(
    {
        "holder",
        "socket",
        "connector",
        "receptacle",
        "terminal",
        "clip",
        "tray",
        "retainer",
        "contacts",
        "spring",
        "header",
        "pins",
    }
)


def off_board_source_class(component_class: str) -> tuple[str, ...]:
    """The tokens that mark a demanded class as a power source the board does not carry.

    Distinct from :func:`class_is_not_a_part`, which names a board or wiring fact: this class
    names a real product, just not one that is placed on the board. Empty for every class the
    library can realize (`coin-cell-holder`) and for every class naming the mate the board does
    carry (`battery-connector`, `coin-cell-socket`).
    """
    canonical = str(component_class or "").strip().casefold().replace("_", "-")
    tokens = _class_tokens(canonical)
    if not (tokens & _OFF_BOARD_SOURCE_TOKENS) or (tokens & _SOURCE_MATE_TOKENS):
        return ()
    if realizable_physical_features(canonical) or reviewed_class_variants(canonical):
        return ()
    return tuple(sorted(tokens & _OFF_BOARD_SOURCE_TOKENS))


def board_outline_fabrication_feature(component_class: str) -> str | None:
    """Return a physical row's board-outline feature only when that reading is proven.

    `shape` alone is not enough: a part can have a shape. The class must explicitly name the
    board/PCB and an outline or shaped-board relation, and no reviewed class or reviewed spelling
    may realize it. This keeps `mounting-hole` and every realizable component demand physical.
    """
    canonical = str(component_class or "").strip().casefold().replace("_", "-")
    tokens = _class_tokens(canonical)
    if not ({"board", "pcb"} & tokens) or not ({"outline", "shaped"} & tokens):
        return None
    if realizable_physical_features(canonical) or reviewed_class_variants(canonical):
        return None
    return canonical


def reviewed_class_variants(component_class: str) -> tuple[str, ...]:
    """Reviewed class names the demanded class is a longer spelling of, best first.

    Only a superset relation is safe to act on: "smt-voltage-regulator" contains the
    reviewed "voltage-regulator", and "stacking-through-hole-header" contains
    "stacking-header". A bare shared token is deliberately NOT a relation —
    "gps-module" shares only "module" with "wifi-module", and renaming a new category to
    a reviewed neighbour would trade a visible late failure for a silently wrong part.
    """
    demanded = _class_tokens(component_class)
    if not demanded:
        return ()
    scored = sorted(
        (
            (-len(tokens), feature)
            for feature in reviewed_feature_vocabulary()
            if (tokens := _class_tokens(feature)) and tokens < demanded
        )
    )
    return tuple(feature for _, feature in scored[:6])


def physical_inventory_record(
    *,
    mpn: str | None,
    symbol: str | None,
    footprint: str | None,
    datasheet: str | None = None,
    sourcing_note: str | None = None,
) -> ReviewedPart | None:
    """Resolve a physical class only from an exact, evidence-backed pair.

    ``datasheet`` and ``sourcing_note`` are intentionally ignored: they can
    aid human review but cannot classify a part.  An explicit MPN takes
    precedence and MUST match an exact reviewed or standard-library identity;
    it never falls back to a symbol/footprint family.
    """
    del datasheet, sourcing_note
    normalized_mpn = (mpn or "").strip().casefold()
    exact_symbol = (symbol or "").strip()
    exact_footprint = (footprint or "").strip()
    if not exact_symbol or not exact_footprint:
        return None
    if normalized_mpn:
        record = reviewed_part(reviewed_identity_for_order_code(normalized_mpn))
        if record is None:
            record = next(
                (
                    candidate
                    for candidate in _STANDARD_LIBRARY_PARTS
                    if candidate.identity == normalized_mpn
                ),
                None,
            )
        if record is None:
            return None
        return (
            record
            if record.symbol == exact_symbol and record.footprint == exact_footprint
            else None
        )
    return _stock_library_physical_record(exact_symbol, exact_footprint)


def reviewed_parts_for_feature(feature: str) -> tuple[ReviewedPart, ...]:
    """Portable reviewed parts implementing one exact physical feature."""
    key = feature.strip().casefold()
    return tuple(
        part for part in reviewed_inventory()
        if part.is_portable_candidate and key in part.physical_features
    )


# Manufacturer evidence reviewed 2026-09-11. These are explicit memberships,
# not interchangeable-device claims and not rules for decoding other suffixes.
# ADI MAX485 ordering table: MAX485ESA+ and MAX485ESA+T, 8-lead narrow SO;
# +T changes the shipping carrier, not the package or device.
# https://www.analog.com/en/products/max485.html
# Nordic nRF52840 Product Specification, Ordering information: QIAA is the
# aQFN-73 variant. R and R7 are separately reviewed order codes; R7 specifies
# a 7-inch reel and MUST NOT be inferred from another suffix.
# https://docs.nordicsemi.com/r/bundle/ps_nrf52840/page/ordering_info.html
# TI ULN2003A orderable device: ULN2003ADR, D/SOIC-16, large tape-and-reel.
# ULN2003 is the brief's unqualified device-series designation, not ULN2004.
# https://www.ti.com/product/ULN2003A/part-details/ULN2003ADR
# Microchip DS21952A, Product Identification System (p. 45), example (b):
# MCP23017-E/SO, extended-temperature 28-lead SOIC (not SPI MCP23S17).
# https://ww1.microchip.com/downloads/en/DeviceDoc/21952a.pdf
# Microchip ATtiny402 ordering information: SSN and SSNR are the same
# SOIC150, -40..105 C, 1.8–5.5 V device grade; R is tape-and-reel only.
# SSFR has a different speed/voltage grade and is intentionally excluded.
# https://onlinedocs.microchip.com/oxy/GUID-5A56DB3A-31E1-4F46-984F-39186535C84E-en-US-7/GUID-CC06B137-262F-43C8-90FD-A14C59BF9C29.html

_DEVICE_MEMBERS: dict[str, frozenset[str]] = {
    # DreamLNK's product page states DL-RFM95 is built on the Semtech SX1276 and
    # publishes the same 16-contact module interface (SPI, RESET, DIO0-DIO5, ANT,
    # 3.3 V, GND); the module is the reviewed physical realization of the
    # brief's unqualified "SX1276 module". Other SX1276 carriers and bare
    # SX1276 order codes are NOT members until separately reviewed.
    # https://www.dreamlnk.com/en/DL-RFM95.html
    "sx1276": frozenset({"dl-rfm95-868m"}),
    "max485": frozenset({"max485esa+", "max485esa+t"}),
    "max485esa+": frozenset({"max485esa+t"}),
    "nrf52840": frozenset({"nrf52840-qiaa-r", "nrf52840-qiaa-r7"}),
    "attiny402": frozenset({"attiny402-ssn", "attiny402-ssnr"}),
    "tps5430": frozenset({"tps5430dda", "tps5430ddar"}),
    # TI's device is the ULN2003A; "ULN2003" is the brief's unqualified series
    # designation (see the sourcing comment above). A group naming either the
    # series or the A-variant is the same reviewed device, so both own the
    # requirement; the curated bundle still supplies the orderable identity.
    "uln2003": frozenset({"uln2003adr", "uln2003a"}),
    "uln2003a": frozenset({"uln2003a", "uln2003adr"}),
    # Microchip DS21952A orders MCP23017 in more than one package.  Only the
    # 28-lead SOIC order code is the reviewed member of this device relation;
    # the SSOP-28 MCP23017T-E/SS record above is a different package and is
    # deliberately NOT a member, so it can never satisfy an "MCP23017" request.
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
    # Reviewed 2026-09-16.  Each member below exists in REVIEWED_PARTS with its
    # own exact symbol/footprint pair, so a device request resolves to concrete
    # hardware instead of a family label.  Family self-equality stays false:
    # these keys name devices, and matches_part_identity still refuses a family
    # label as construction evidence.
    "max31855": frozenset({"max31855kasa+"}),
    "rp2040": frozenset({"rp2040"}),
    "sn65hvd230": frozenset({"sn65hvd230"}),
    "pca9685": frozenset({"pca9685pw,118"}),
    "esp32-s3": frozenset({"esp32-s3-wroom-1-n8r8", "esp32-s3-wroom-1-n16r8"}),
    "esp32-c3": frozenset({"esp32-c3-mini-1-n4"}),
    "ads1115": frozenset({"ads1115idgsr"}),
    "drv8833": frozenset({"drv8833pwpr"}),
    # ME6211C33M5G-N and TLV62569DBVR are the exact order codes the reviewed
    # me6211-3v3@1 and tlv62569-3v3@1 recipes already emit for their published
    # families; AP2112K-3.3TRG1 is the 3.3 V order code of the AP2112K device.
    "me6211-3v3": frozenset({"me6211c33m5g-n"}),
    "tlv62569": frozenset({"tlv62569dbvr"}),
    "tlv62569-3v3": frozenset({"tlv62569dbvr"}),
    "ap2112k-3.3": frozenset({"ap2112k-3.3trg1"}),
    # Recipe-emitted identities (2026-09-16).  Each member is a record above;
    # a recipe that names the device or the module series resolves to the exact
    # order code its own data sheet documents.
    "attiny1614": frozenset({"attiny1614-ssnr"}),
    "attiny412": frozenset({"attiny412-ssn"}),
    "ch32v003": frozenset({"ch32v003j4m6"}),
    "a4988": frozenset({"a4988settr-t"}),
    "ws2812b": frozenset({"ws2812b-b/t"}),
    "esp32-s3-mini-1": frozenset({"esp32-s3-mini-1-n8"}),
    "esp32-wroom-32e": frozenset({"esp32-wroom-32e-n4"}),
    # "TL3342" is E-Switch's switch series, not an order code: the member is the
    # reviewed TL3342F260QG record (5.2 x 5.2 mm SPST-NO tactile switch).  The
    # stock kicad-tl3342-button id is a land-pattern contract with no
    # manufacturer identity and is deliberately NOT a member, so a device-level
    # "TL3342" requirement can never be satisfied by it.
    "tl3342": frozenset({"tl3342f260qg"}),
}


# ST's STM32L031K6 product page lists STM32L031K6T6 in the STM32L0 series;
# ordering scheme: K=32 pins, 6=32-Kbyte flash, T=LQFP, 6=-40..85 Celsius.
# STM32L072CZU6 is independently reviewed as a UFQFPN-48 package and is NOT
# package-equivalent to the LQFP candidates. No inference is made for other L0
# devices, densities, or package codes.
# https://www.st.com/en/microcontrollers-microprocessors/stm32l031k6.html
# https://www.st.com/resource/en/datasheet/stm32l031k4.pdf
# https://www.st.com/resource/en/datasheet/stm32l072cz.pdf
_FAMILY_MEMBERS: dict[str, frozenset[str]] = {
    "stm32l0": frozenset({"stm32l031k6t6", "stm32l072czu6"}),
    # "STM32F103" and "STM32" are design-level family selectors, never
    # construction hardware: every member below is an independently reviewed
    # STM32 identity.  stm32l072czu6 is reviewed identity-only (no library
    # pair), so it can satisfy an identity request but can never be selected
    # for a BOM.  No package, density, or order code is inferred for any other
    # STM32 device or for the family label itself.
    "stm32f103": frozenset({"stm32f103c8t6"}),
    "stm32": frozenset({"stm32f103c8t6", "stm32l031k6t6", "stm32l072czu6"}),
}


def accepted_part_identities(family: str) -> tuple[str, ...]:
    """Exact reviewed concrete identities accepted for a family/device key.

    The tuple is sorted for deterministic feedback. Unknown keys return no
    candidates rather than widening through punctuation or name prefixes.
    """
    key = family.strip().casefold()
    return tuple(sorted(_FAMILY_MEMBERS.get(key, _DEVICE_MEMBERS.get(key, ()))))


def is_part_family(identity: str) -> bool:
    """Whether this exact identity is a reviewed non-device family selector."""
    return identity.strip().casefold() in _FAMILY_MEMBERS


def declares_package(declared: str, reviewed: str) -> bool:
    """Whether a reviewed package/footprint description declares this package.

    The comparison is token-exact: the declared designation (for example
    ``LQFP-48``) must appear as a whole delimited token in the reviewed
    description, so ``LQFP-48 7x7mm 0.50mm`` or a library footprint path
    ``LQFP-48_L7.0-...`` satisfy it while ``LQFP-144`` or ``LQFP-48X`` do not.
    A reviewed record is still required independently; this never widens the
    accepted identity set.
    """
    declared_key = declared.strip().casefold()
    if not declared_key:
        return False
    tokens = {token for token in re.split(r"[^0-9a-z+.\-]+", reviewed.strip().casefold()) if token}
    return declared_key in tokens


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
