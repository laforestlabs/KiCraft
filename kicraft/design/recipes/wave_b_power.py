"""Wave-B exact power-entry and regulation recipes.

Safety-critical entries remain canary until the required independent human review.
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

_DATE = "2026-09-09"
_C = ("Device:C", "Capacitor_SMD:C_0603_1608Metric")
_R = ("Device:R", "Resistor_SMD:R_0603_1608Metric")
_L = ("Device:L", "Inductor_SMD:L_6.3x6.3_H3")


def _part(
    role: str, prefix: str, value: str, symbol: str, footprint: str, *, mpn: str | None = None
) -> Group:
    return Group(
        role=role,
        reference_prefix=prefix,
        value=value,
        symbol=symbol,
        footprint=footprint,
        sheet_role="power",
        mpn=mpn,
    )


def _passive(role: str, prefix: str, value: str, quantity: int = 1) -> Group:
    symbol, footprint = _C if prefix == "C" else _R if prefix == "R" else _L
    return _part(role, prefix, value, symbol, footprint).model_copy(update={"quantity": quantity})


def _source(url: str, title: str, *sections: str) -> tuple[Source, ...]:
    return (
        Source(url=url, title=title, revision="current", reviewed_date=_DATE, sections=sections),
    )


def _power_recipe(
    *,
    recipe: str,
    family: str,
    exact_part: str,
    parts: tuple[Group, ...],
    pins: tuple[Pin, ...],
    ports: tuple[Port, ...],
    internal_nets: tuple[str, ...],
    source_url: str,
    assertions: tuple[Assertion, ...],
    no_connects: tuple[NoConnect, ...] = (),
    parameters: dict | None = None,
    allowed: dict | None = None,
    protected: tuple[str, ...] = (),
) -> RecipeDefinition:
    return RecipeDefinition(
        recipe=recipe,
        family=family,
        exact_part=exact_part,
        maturity="production",
        protected_aliases=(family, exact_part, *protected),
        identity_aliases=(exact_part,),
        required_sheet_roles=("power",),
        parameter_defaults=parameters or {},
        allowed_parameters=allowed or {},
        ports=ports,
        internal_nets=internal_nets,
        parts=parts,
        pins=pins,
        no_connects=no_connects,
        placement_constraints=(
            Placement(
                kind="power_loop", role=parts[0].role, parameters={"minimize_loop_area": True}
            ),
            Placement(kind="decoupling_proximity", role=parts[0].role, parameters={"max_mm": 3.0}),
        ),
        electrical_assertions=assertions,
        source_documents=_source(
            source_url,
            f"{exact_part} reviewed application data",
            "Typical Application",
            "Electrical Characteristics",
            "Layout Guidelines",
        ),
    )


USB_C_5V_SINK = _power_recipe(
    recipe="usb-c-5v-sink@1",
    family="usb-c-power-sink",
    exact_part="USB-C-5V-SINK",
    protected=("TYPE-C-31-M-12",),
    ports=(
        Port(name="vbus", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="shield", direction="passive", required=False, allow_ground=True),
    ),
    internal_nets=("cc1", "cc2", "shield"),
    parts=(
        _part(
            "connector",
            "J",
            "TYPE-C-31-M-12",
            "usb-c-16p:TYPE-C-31-M-12",
            "usb-c-16p:USB-C_SMD-TYPE-C-31-M-12_1",
            mpn="TYPE-C-31-M-12",
        ),
        _passive("cc_pulldown", "R", "5.1k", 2),
    ),
    pins=(
        Pin(role="connector", pin="A4B9", net="vbus"),
        Pin(role="connector", pin="B4A9", net="vbus"),
        Pin(role="connector", pin="A1B12", net="gnd"),
        Pin(role="connector", pin="B1A12", net="gnd"),
        Pin(role="connector", pin="A5", net="cc1"),
        Pin(role="connector", pin="B5", net="cc2"),
        Pin(role="cc_pulldown", index=0, pin="1", net="cc1"),
        Pin(role="cc_pulldown", index=0, pin="2", net="gnd"),
        Pin(role="cc_pulldown", index=1, pin="1", net="cc2"),
        Pin(role="cc_pulldown", index=1, pin="2", net="gnd"),
        *(Pin(role="connector", pin=str(pin), net="shield") for pin in range(1, 5)),
    ),
    no_connects=tuple(
        NoConnect(role="connector", pin=pin) for pin in ("A6", "A7", "B6", "B7", "A8", "B8")
    ),
    source_url="https://www.usb.org/document-library/usb-type-cr-cable-and-connector-specification-release-24",
    assertions=(
        Assertion(
            code="usb_c_sink_cc_resistors",
            message="CC1 and CC2 each have an independent 5.1 kOhm Rd to ground",
        ),
    ),
)

USB_C_USB2_DEVICE = _power_recipe(
    recipe="usb-c-usb2-device@1",
    family="usb-c-usb2-device",
    exact_part="USB-C-USB2-DEVICE",
    protected=("TYPE-C-31-M-12", "USBLC6-2SC6"),
    parameters={"series_resistors": True},
    allowed={"series_resistors": (False, True)},
    ports=(
        Port(name="vbus", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="usb_dp", direction="bidirectional"),
        Port(name="usb_dm", direction="bidirectional"),
        Port(name="shield", direction="passive", required=False, allow_ground=True),
    ),
    internal_nets=("cc1", "cc2", "dp_conn", "dm_conn", "dp_esd", "dm_esd", "shield"),
    parts=(
        _part(
            "connector",
            "J",
            "TYPE-C-31-M-12",
            "usb-c-16p:TYPE-C-31-M-12",
            "usb-c-16p:USB-C_SMD-TYPE-C-31-M-12_1",
            mpn="TYPE-C-31-M-12",
        ),
        _passive("cc_pulldown", "R", "5.1k", 2),
        _part(
            "esd",
            "U",
            "USBLC6-2SC6",
            "usblc6-2sc6:USBLC6-2SC6_C2687116",
            "usblc6-2sc6:SOT-23-6_L2.9-W1.6-P0.95-LS2.8-BL",
            mpn="USBLC6-2SC6",
        ),
        _passive("usb_series", "R", "22R", 2),
    ),
    pins=(
        Pin(role="connector", pin="A4B9", net="vbus"),
        Pin(role="connector", pin="B4A9", net="vbus"),
        Pin(role="connector", pin="A1B12", net="gnd"),
        Pin(role="connector", pin="B1A12", net="gnd"),
        Pin(role="connector", pin="A5", net="cc1"),
        Pin(role="connector", pin="B5", net="cc2"),
        Pin(role="cc_pulldown", index=0, pin="1", net="cc1"),
        Pin(role="cc_pulldown", index=0, pin="2", net="gnd"),
        Pin(role="cc_pulldown", index=1, pin="1", net="cc2"),
        Pin(role="cc_pulldown", index=1, pin="2", net="gnd"),
        Pin(role="connector", pin="A6", net="dp_conn"),
        Pin(role="connector", pin="B6", net="dp_conn"),
        Pin(role="connector", pin="A7", net="dm_conn"),
        Pin(role="connector", pin="B7", net="dm_conn"),
        Pin(role="esd", pin="1", net="dm_conn"),
        Pin(role="esd", pin="6", net="dp_conn"),
        Pin(role="esd", pin="2", net="gnd"),
        Pin(role="esd", pin="5", net="vbus"),
        Pin(role="esd", pin="3", net="dm_esd"),
        Pin(role="esd", pin="4", net="dp_esd"),
        Pin(role="usb_series", index=0, pin="1", net="dm_esd"),
        Pin(role="usb_series", index=0, pin="2", net="usb_dm"),
        Pin(role="usb_series", index=1, pin="1", net="dp_esd"),
        Pin(role="usb_series", index=1, pin="2", net="usb_dp"),
        *(Pin(role="connector", pin=str(pin), net="shield") for pin in range(1, 5)),
    ),
    no_connects=(NoConnect(role="connector", pin="A8"), NoConnect(role="connector", pin="B8")),
    source_url="https://www.usb.org/document-library/usb-type-cr-cable-and-connector-specification-release-24",
    assertions=(
        Assertion(
            code="usb2_esd_and_cc",
            message="USB D+/D- have connector-side ESD protection and both CC pins have Rd",
        ),
    ),
)

CH224K_PD_TRIGGER = _power_recipe(
    recipe="ch224k-pd-trigger@1",
    family="usb-pd-fixed-trigger",
    exact_part="CH224K",
    protected=("TYPE-C-31-M-12",),
    ports=(
        Port(name="vbus", direction="power"),
        Port(name="gnd", direction="power"),
    ),
    internal_nets=("cc1", "cc2", "shield", "vdd", "vbus_sense", "cfg1", "cfg2", "cfg3", "pg"),
    parameters={"output_voltage": 9},
    allowed={"output_voltage": (9,)},
    parts=(
        _part(
            "connector",
            "J",
            "TYPE-C-31-M-12",
            "usb-c-16p:TYPE-C-31-M-12",
            "usb-c-16p:USB-C_SMD-TYPE-C-31-M-12_1",
            mpn="TYPE-C-31-M-12",
        ),
        _part(
            "controller",
            "U",
            "CH224K",
            "ch224k:CH224K",
            "ch224k:ESSOP-10_L4.9-W3.9-P1.0-LS6.0-TL-EP",
            mpn="CH224K",
        ),
        _passive("config", "R", "10k", 3),
        _passive("vdd_feed", "R", "1k"),
        _passive("vbus_sense", "R", "10k"),
        _passive("decoupling", "C", "1uF"),
    ),
    pins=(
        Pin(role="connector", pin="A4B9", net="vbus"),
        Pin(role="connector", pin="B4A9", net="vbus"),
        Pin(role="connector", pin="A1B12", net="gnd"),
        Pin(role="connector", pin="B1A12", net="gnd"),
        Pin(role="connector", pin="A5", net="cc1"),
        Pin(role="connector", pin="B5", net="cc2"),
        *(Pin(role="connector", pin=str(pin), net="shield") for pin in range(1, 5)),
        Pin(role="controller", pin="1", net="vdd"),
        Pin(role="controller", pin="8", net="vbus_sense"),
        Pin(role="controller", pin="11", net="gnd"),
        Pin(role="controller", pin="7", net="cc1"),
        Pin(role="controller", pin="6", net="cc2"),
        Pin(role="controller", pin="9", net="cfg1"),
        Pin(role="controller", pin="2", net="cfg2"),
        Pin(role="controller", pin="3", net="cfg3"),
        *(
            Pin(role="config", index=i, pin="1", net=net)
            for i, net in enumerate(("cfg1", "cfg2", "cfg3"))
        ),
        *(Pin(role="config", index=i, pin="2", net="gnd") for i in range(3)),
        Pin(role="vdd_feed", pin="1", net="vbus"),
        Pin(role="vdd_feed", pin="2", net="vdd"),
        Pin(role="vbus_sense", pin="1", net="vbus"),
        Pin(role="vbus_sense", pin="2", net="vbus_sense"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
    ),
    no_connects=(
        *(NoConnect(role="connector", pin=pin) for pin in ("A6", "A7", "B6", "B7", "A8", "B8")),
        NoConnect(role="controller", pin="4"),
        NoConnect(role="controller", pin="5"),
        NoConnect(role="controller", pin="10"),
    ),
    source_url="https://www.wch-ic.com/downloads/CH224DS1_PDF.html",
    assertions=(
        Assertion(
            code="pd_fixed_pdo_review",
            message="9 V level selection is CFG1=CFG2=CFG3=0; VDD and VBUS sense use datasheet series resistors",
        ),
    ),
)


# WCH 1F sections 5.2.1/6.1 specify resistor selection, not three logic outputs.
# The SS13D07VG4 manufacturer drawing has a long common contact (pad 2)
# selecting pads 1, 3, 4. Its installed manifest's "SPDT/3-pin" prose is wrong;
# the installed symbol/footprint correctly include four contacts and two tabs.
CH224K_PD_SELECTABLE = RecipeDefinition(
    recipe="ch224k-pd-selectable@1",
    family="usb-pd-selectable-trigger",
    default_for_family=True,
    maturity="production",
    protected_aliases=("usb-pd-selectable-trigger",),
    required_sheet_roles=("power",),
    parameter_defaults={
        "voltage_options": "9/12/20V",
        "selection_mode": "resistor-sp3t",
        "selector_part": "SS13D07VG4",
    },
    allowed_parameters={
        "voltage_options": ("9/12/20V",),
        "selection_mode": ("resistor-sp3t",),
        "selector_part": ("SS13D07VG4",),
    },
    ports=(
        Port(name="cc1", direction="bidirectional"),
        Port(name="cc2", direction="bidirectional"),
        Port(name="vbus", direction="power"),
        Port(name="gnd", direction="power"),
    ),
    internal_nets=("vdd", "vbus_sense", "pd_only", "cfg1", "select_9v", "select_12v"),
    parts=(
        _part(
            "controller",
            "U",
            "CH224K",
            "ch224k:CH224K",
            "ch224k:ESSOP-10_L4.9-W3.9-P1.0-LS6.0-TL-EP",
            mpn="CH224K",
        ),
        _part(
            "selector",
            "SW",
            "SS13D07VG4",
            "ss13d07vg4:SS13D07VG4",
            "ss13d07vg4:SW-TH_SS13D07VG4",
            mpn="SS13D07VG4",
        ),
        _passive("select_9v", "R", "6.8k 1%"),
        _passive("select_12v", "R", "24k 1%"),
        # At 22 V input and 3.24 V VDD this dissipates 0.352 W nominal.
        _part(
            "vdd_feed",
            "R",
            "1k 1% 0.5W",
            "Device:R",
            "Resistor_SMD:R_1210_3225Metric",
        ),
        _passive("vbus_sense", "R", "10k"),
        _passive("decoupling", "C", "1uF 10V"),
    ),
    pins=(
        Pin(role="controller", pin="1", net="vdd"),
        Pin(role="controller", pin="4", net="pd_only"),
        Pin(role="controller", pin="5", net="pd_only"),
        Pin(role="controller", pin="6", net="cc2"),
        Pin(role="controller", pin="7", net="cc1"),
        Pin(role="controller", pin="8", net="vbus_sense"),
        Pin(role="controller", pin="9", net="cfg1"),
        Pin(role="controller", pin="11", net="gnd"),
        Pin(role="selector", pin="1", net="select_9v"),
        Pin(role="selector", pin="2", net="cfg1"),
        Pin(role="selector", pin="3", net="select_12v"),
        Pin(role="selector", pin="5", net="gnd"),
        Pin(role="selector", pin="6", net="gnd"),
        Pin(role="select_9v", pin="1", net="select_9v"),
        Pin(role="select_9v", pin="2", net="gnd"),
        Pin(role="select_12v", pin="1", net="select_12v"),
        Pin(role="select_12v", pin="2", net="gnd"),
        Pin(role="vdd_feed", pin="1", net="vbus"),
        Pin(role="vdd_feed", pin="2", net="vdd"),
        Pin(role="vbus_sense", pin="1", net="vbus"),
        Pin(role="vbus_sense", pin="2", net="vbus_sense"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
    ),
    no_connects=(
        *(NoConnect(role="controller", pin=pin) for pin in ("2", "3", "10")),
        NoConnect(role="selector", pin="4"),
    ),
    placement_constraints=(
        Placement(kind="decoupling_proximity", role="controller", parameters={"max_mm": 3.0}),
    ),
    electrical_assertions=(
        Assertion(
            code="pd_resistor_selector",
            message="SS13D07VG4 common pin 2 selects pin 1: 6.8k to GND requests 9V; "
            "pin 3: 24k requests 12V; pin 4: open requests 20V. CFG2/CFG3 remain NC.",
        ),
        Assertion(
            code="pd_direct_cc",
            message="Connect CC1/CC2 directly to a USB-C receptacle, with no parallel "
            "external Rd or separate 5V sink. CH224K provides the sink termination. "
            "DP/DM are shorted locally for PD-only operation, not connected to USB data.",
        ),
        Assertion(
            code="pd_selector_assembly",
            message="Populate both 1% selection resistors and a >=0.5W 1k VDD feed "
            "resistor in 1210. Solder exposed ground pad and both switch frame tabs to GND. "
            "Label positions by continuity (2-1=9V, 2-3=12V, 2-4=20V). "
            "Select with power disconnected; contact gaps request 20V. The source must "
            "offer the requested PDO; VBUS starts at 5V and is not a regulated output. "
            "All external VBUS circuitry and the load must tolerate 20V plus transients. "
            "This receptacle block does not implement E-Mark simulation or guarantee >3A.",
        ),
    ),
    source_documents=(
        Source(
            url="https://components101.com/sites/default/files/component_datasheet/WCH_CH224K_ENG.pdf",
            title="WCH CH224 manual (English translation)",
            revision="1F",
            reviewed_date="2026-09-11",
            sections=(
                "4.3 pin functions",
                "5.2.1 resistor configuration",
                "5.5 PD only",
                "6.1 CH224K female-port reference schematic",
                "7.5 VDD regulator",
            ),
        ),
        Source(
            url="https://datasheet.lcsc.com/datasheet/pdf/39fcef34462917ff9922c33e708581d0.pdf?productCode=C2681578",
            title="SHOUHAN SS-13D07VG4 specification for approval",
            revision="2017-02-15 / drawing 2016-04-02",
            reviewed_date="2026-09-11",
            sections=("page 4 contact schematic and PCB layout",),
        ),
    ),
)

TP4056_1S_CHARGER = _power_recipe(
    recipe="tp4056-1s-charger@1",
    family="single-cell-liion-charger",
    exact_part="TP4056",
    ports=(
        Port(name="input", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="battery", direction="power"),
        Port(name="charge_status", direction="output", required=False),
        Port(name="standby_status", direction="output", required=False),
    ),
    internal_nets=("prog",),
    parameters={"charge_current_ma": 1000},
    allowed={"charge_current_ma": (1000,)},
    parts=(
        _part(
            "charger",
            "U",
            "TP4056",
            "tp4056:TP4056_C725790",
            "tp4056:ESOP-8_L4.9-W3.9-P1.27-LS6.0-BL-EP",
            mpn="TP4056",
        ),
        _passive("program_resistor", "R", "1.2k"),
        _passive("input_cap", "C", "10uF"),
        _passive("battery_cap", "C", "10uF"),
    ),
    pins=(
        Pin(role="charger", pin="4", net="input"),
        Pin(role="charger", pin="3", net="gnd"),
        Pin(role="charger", pin="9", net="gnd"),
        Pin(role="charger", pin="1", net="gnd"),
        Pin(role="charger", pin="5", net="battery"),
        Pin(role="charger", pin="2", net="prog"),
        Pin(role="charger", pin="7", net="charge_status"),
        Pin(role="charger", pin="6", net="standby_status"),
        Pin(role="charger", pin="8", net="input"),
        Pin(role="program_resistor", pin="1", net="prog"),
        Pin(role="program_resistor", pin="2", net="gnd"),
        Pin(role="input_cap", pin="1", net="input"),
        Pin(role="input_cap", pin="2", net="gnd"),
        Pin(role="battery_cap", pin="1", net="battery"),
        Pin(role="battery_cap", pin="2", net="gnd"),
    ),
    source_url="https://www.toppwr.com/uploadfile/file/20241228/676f612b9f2dd.pdf",
    assertions=(
        Assertion(
            code="charger_thermal_limit",
            message="Charge current and copper thermal area must satisfy the reviewed TP4056 dissipation bound",
        ),
        Assertion(
            code="single_cell_only", message="Battery port is one protected Li-ion/LiPo cell only"
        ),
    ),
)


def _three_pin_ldo(
    *,
    recipe: str,
    family: str,
    part: str,
    symbol: str,
    footprint: str,
    datasheet: str,
    vin: str,
    gnd: str,
    vout: str,
    cin: str,
    cout: str,
    extra_nc: str | None = None,
    enable_pin: str | None = None,
) -> RecipeDefinition:
    no_connects = (NoConnect(role="regulator", pin=extra_nc),) if extra_nc else ()
    enable = (Pin(role="regulator", pin=enable_pin, net="input"),) if enable_pin else ()
    return _power_recipe(
        recipe=recipe,
        family=family,
        exact_part=part,
        ports=(
            Port(name="input", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="output", direction="power"),
        ),
        internal_nets=(),
        parts=(
            _part("regulator", "U", part, symbol, footprint, mpn=part),
            _passive("input_cap", "C", cin),
            _passive("output_cap", "C", cout),
        ),
        pins=(
            Pin(role="regulator", pin=vin, net="input"),
            Pin(role="regulator", pin=gnd, net="gnd"),
            Pin(role="regulator", pin=vout, net="output"),
            *enable,
            Pin(role="input_cap", pin="1", net="input"),
            Pin(role="input_cap", pin="2", net="gnd"),
            Pin(role="output_cap", pin="1", net="output"),
            Pin(role="output_cap", pin="2", net="gnd"),
        ),
        no_connects=no_connects,
        source_url=datasheet,
        assertions=(
            Assertion(
                code="ldo_stability_caps",
                message=f"Input {cin} and output {cout} capacitors satisfy the exact regulator stability requirements",
            ),
        ),
    )


ME6211_3V3 = _three_pin_ldo(
    recipe="me6211-3v3@1",
    family="me6211-3v3",
    part="ME6211C33M5G-N",
    symbol="me6211c33:ME6211C33M5G-N",
    footprint="me6211c33:SOT-23-5_L3.0-W1.7-P0.95-LS2.8-BL",
    datasheet="https://www.lcsc.com/datasheet/C82942.pdf",
    vin="1",
    gnd="2",
    vout="5",
    cin="1uF",
    cout="1uF",
    extra_nc="4",
    enable_pin="3",
)
MCP1700_3V3 = _three_pin_ldo(
    recipe="mcp1700-3v3@1",
    family="mcp1700-3v3",
    part="MCP1700T-3302E/TT",
    symbol="Regulator_Linear:MCP1700x-330xxTT",
    footprint="Package_TO_SOT_SMD:SOT-23",
    datasheet="https://ww1.microchip.com/downloads/aemDocuments/documents/APID/ProductDocuments/DataSheets/MCP1700-Low-Quiescent-Current-LDO-20001826F.pdf",
    vin="3",
    gnd="1",
    vout="2",
    cin="1uF",
    cout="1uF",
)
AMS1117_3V3 = _three_pin_ldo(
    recipe="ams1117-3v3@1",
    family="ams1117-3v3",
    part="AMS1117-3.3",
    symbol="Regulator_Linear:AMS1117-3.3",
    footprint="Package_TO_SOT_SMD:SOT-223-3_TabPin2",
    datasheet="https://www.advanced-monolithic.com/pdf/ds1117.pdf",
    vin="3",
    gnd="1",
    vout="2",
    cin="10uF",
    cout="22uF",
)

MCP6001_FOLLOWER = _power_recipe(
    recipe="mcp6001-follower@1",
    family="mcp6001-follower",
    exact_part="MCP6001T-I/OT",
    ports=(
        Port(name="vdd", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="input", direction="input"),
        Port(name="output", direction="output"),
    ),
    internal_nets=(),
    parts=(
        _part(
            "amplifier",
            "U",
            "MCP6001T-I/OT",
            "mcp6001:MCP6001T-I_OT",
            "mcp6001:SOT-23-5_L3.0-W1.7-P0.95-LS2.8-BR",
            mpn="MCP6001T-I/OT",
        ),
        _passive("decoupling", "C", "100nF"),
    ),
    pins=(
        Pin(role="amplifier", pin="5", net="vdd"),
        Pin(role="amplifier", pin="2", net="gnd"),
        Pin(role="amplifier", pin="3", net="input"),
        Pin(role="amplifier", pin="1", net="output"),
        Pin(role="amplifier", pin="4", net="output"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
    ),
    source_url="https://ww1.microchip.com/downloads/aemDocuments/documents/MSLD/ProductDocuments/DataSheets/MCP6001-1R-1U-2-4-Low-Power-Op-Amp-DS20001733L.pdf",
    assertions=(
        Assertion(
            code="follower_feedback",
            message="Inverting input is connected directly to output for unity gain",
        ),
    ),
)


def _buck(
    *,
    recipe: str,
    family: str,
    part: str,
    symbol: str,
    footprint: str,
    datasheet: str,
    pins_map: dict[str, str],
    output_voltage: float,
    inductor: str = "4.7uH",
    feedback_top: str = "100k",
    feedback_bottom: str = "22k",
    fixed_output: bool = False,
    enable_divider: bool = False,
) -> RecipeDefinition:
    parts = [
        _part("converter", "U", part, symbol, footprint, mpn=part),
        _passive("inductor", "L", inductor),
        _passive("input_cap", "C", "10uF", 2),
        _passive("output_cap", "C", "22uF", 2),
        _passive("feedback_top", "R", feedback_top),
        _passive("feedback_bottom", "R", feedback_bottom),
    ]
    pins = [
        Pin(role="converter", pin=pins_map["vin"], net="input"),
        Pin(role="converter", pin=pins_map["gnd"], net="gnd"),
        Pin(role="converter", pin=pins_map["sw"], net="switch"),
        Pin(role="converter", pin=pins_map["fb"], net="feedback"),
        Pin(role="inductor", pin="1", net="switch"),
        Pin(role="inductor", pin="2", net="output"),
        *(Pin(role="input_cap", index=i, pin="1", net="input") for i in range(2)),
        *(Pin(role="input_cap", index=i, pin="2", net="gnd") for i in range(2)),
        *(Pin(role="output_cap", index=i, pin="1", net="output") for i in range(2)),
        *(Pin(role="output_cap", index=i, pin="2", net="gnd") for i in range(2)),
        Pin(role="feedback_top", pin="1", net="output"),
        Pin(role="feedback_top", pin="2", net="feedback"),
        Pin(role="feedback_bottom", pin="1", net="feedback"),
        Pin(role="feedback_bottom", pin="2", net="gnd"),
    ]
    internal_nets = ["switch", "feedback"]
    if pin := pins_map.get("en"):
        if enable_divider:
            internal_nets.append("enable")
            parts.extend(
                (
                    _passive("enable_top", "R", "511k"),
                    _passive("enable_bottom", "R", "100k"),
                )
            )
            pins.extend(
                (
                    Pin(role="converter", pin=pin, net="enable"),
                    Pin(role="enable_top", pin="1", net="input"),
                    Pin(role="enable_top", pin="2", net="enable"),
                    Pin(role="enable_bottom", pin="1", net="enable"),
                    Pin(role="enable_bottom", pin="2", net="gnd"),
                )
            )
        else:
            pins.append(Pin(role="converter", pin=pin, net="input"))
    if fixed_output:
        parts = [part for part in parts if part.role not in {"feedback_top", "feedback_bottom"}]
        pins = [
            pin.model_copy(update={"net": "output"})
            if pin.role == "converter" and pin.pin == pins_map["fb"]
            else pin
            for pin in pins
            if pin.role not in {"feedback_top", "feedback_bottom"}
        ]
    if pin := pins_map.get("ep"):
        pins.append(Pin(role="converter", pin=pin, net="gnd"))
    if pin := pins_map.get("boot"):
        internal_nets.append("bootstrap")
        parts.append(_passive("bootstrap_cap", "C", "100nF"))
        pins.extend(
            (
                Pin(role="converter", pin=pin, net="bootstrap"),
                Pin(role="bootstrap_cap", pin="1", net="bootstrap"),
                Pin(role="bootstrap_cap", pin="2", net="switch"),
            )
        )
    if pin := pins_map.get("ss"):
        internal_nets.append("softstart")
        parts.append(_passive("softstart_cap", "C", "10nF"))
        pins.extend(
            (
                Pin(role="converter", pin=pin, net="softstart"),
                Pin(role="softstart_cap", pin="1", net="softstart"),
                Pin(role="softstart_cap", pin="2", net="gnd"),
            )
        )
    if pin := pins_map.get("comp"):
        internal_nets.append("compensation")
        parts.extend(
            (_passive("comp_resistor", "R", "6.8k"), _passive("comp_capacitor", "C", "2.2nF"))
        )
        pins.extend(
            (
                Pin(role="converter", pin=pin, net="compensation"),
                Pin(role="comp_resistor", pin="1", net="compensation"),
                Pin(role="comp_resistor", pin="2", net="comp_rc"),
                Pin(role="comp_capacitor", pin="1", net="comp_rc"),
                Pin(role="comp_capacitor", pin="2", net="gnd"),
            )
        )
        internal_nets.append("comp_rc")
    return _power_recipe(
        recipe=recipe,
        family=family,
        exact_part=part,
        ports=(
            Port(name="input", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="output", direction="power"),
        ),
        internal_nets=tuple(internal_nets),
        parameters={"output_voltage": output_voltage},
        allowed={"output_voltage": (output_voltage,)},
        parts=tuple(parts),
        pins=tuple(pins),
        source_url=datasheet,
        assertions=(
            Assertion(
                code="buck_component_bounds",
                message="Inductor saturation, capacitor voltage, feedback tolerance, and thermal limits require independent review",
            ),
        ),
    )


TLV62569_3V3 = _buck(
    recipe="tlv62569-3v3@1",
    family="tlv62569-3v3",
    part="TLV62569DBVR",
    symbol="tlv62569:TLV62569DBVR",
    footprint="tlv62569:SOT-23-5_L3.0-W1.7-P0.95-LS2.8-BR",
    datasheet="https://www.ti.com/lit/ds/symlink/tlv62569.pdf",
    pins_map={"vin": "4", "gnd": "2", "sw": "3", "fb": "5", "en": "1"},
    output_voltage=3.3,
)
AP63203_3V3 = _buck(
    recipe="ap63203-3v3@1",
    family="ap63203-3v3",
    part="AP63203WU-7",
    symbol="ap63203:AP63203WU-7",
    footprint="ap63203:TSOT-26_L2.9-W1.6-P0.95-LS2.8-BL",
    datasheet="https://www.diodes.com/assets/Datasheets/AP63200-AP63201-AP63203-AP63205.pdf",
    pins_map={"vin": "3", "gnd": "4", "sw": "5", "fb": "1", "en": "2", "boot": "6"},
    output_voltage=3.3,
    fixed_output=True,
)
AP63205_5V = _buck(
    recipe="ap63205-5v@1",
    family="ap63205-5v",
    part="AP63205WU-7",
    symbol="ap63205:AP63205WU-7",
    footprint="ap63205:TSOT-23-6_L2.9-W1.6-P0.95-LS2.8-BL",
    datasheet="https://www.diodes.com/assets/Datasheets/AP63200-AP63201-AP63203-AP63205.pdf",
    pins_map={"vin": "3", "gnd": "4", "sw": "5", "fb": "1", "en": "2", "boot": "6"},
    output_voltage=5.0,
    fixed_output=True,
)
TPS54331_ADJUSTABLE = _buck(
    recipe="tps54331-adjustable@1",
    family="tps54331-adjustable",
    part="TPS54331DDAR",
    symbol="tps54331:TPS54331DDAR",
    footprint="tps54331:SOIC-8_L4.9-W3.9-P1.27-LS6.0-BL-EP",
    datasheet="https://www.ti.com/lit/ds/symlink/tps54331.pdf",
    pins_map={
        "vin": "2",
        "gnd": "7",
        "sw": "8",
        "fb": "5",
        "boot": "1",
        "en": "3",
        "ss": "4",
        "comp": "6",
        "ep": "9",
    },
    output_voltage=3.3,
    inductor="15uH",
    feedback_top="68.1k",
    feedback_bottom="22k",
    enable_divider=True,
)


def expand_usb_c_usb2_device(resolved):
    """Drop the connector-owned 22R pair when the MCU already owns it."""
    from .registry import expand_static_definition

    definition = USB_C_USB2_DEVICE
    if not resolved.parameters.get("series_resistors", True):
        definition = definition.model_copy(
            update={
                "internal_nets": tuple(
                    net
                    for net in definition.internal_nets
                    if net not in {"dp_esd", "dm_esd"}
                ),
                "parts": tuple(
                    part for part in definition.parts if part.role != "usb_series"
                ),
                "pins": tuple(
                    (
                        pin.model_copy(update={"net": {"3": "usb_dm", "4": "usb_dp"}[pin.pin]})
                        if pin.role == "esd" and pin.pin in {"3", "4"}
                        else pin
                    )
                    for pin in definition.pins
                    if pin.role != "usb_series"
                ),
            }
        )
    return expand_static_definition(definition, resolved)


WAVE_B_POWER_RECIPES = (
    USB_C_5V_SINK,
    USB_C_USB2_DEVICE,
    CH224K_PD_TRIGGER,
    CH224K_PD_SELECTABLE,
    TP4056_1S_CHARGER,
    ME6211_3V3,
    MCP1700_3V3,
    MCP6001_FOLLOWER,
    AMS1117_3V3,
    TLV62569_3V3,
    AP63203_3V3,
    AP63205_5V,
    TPS54331_ADJUSTABLE,
)
