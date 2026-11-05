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

# Exact JLC/LCSC-stocked passives used in the high-current power recipes.  The
# value string deliberately retains the MPN so a board-artifact observer can
# bind the installed footprint to the reviewed part, rather than infer a
# capacitor's rating from a generic ``Device:C`` symbol.
_CATALOG_CAPACITORS = {
    "10nF 50V X7R": (
        "FCC0603B103K500CT",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8780058373529083904",
    ),
    "100nF 16V X7R": (
        "CL10B104KO8NNNC",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8586178403602587648",
    ),
    "100nF 25V X7R": (
        "GRM188R71E104KA01D",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588932819465228288",
    ),
    "100nF 50V X7R": (
        "CC0603KRX7R9BB104",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8667086713681608704",
    ),
    "47nF 50V X7R": (
        "CL10B473KB8NNNC",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8579707009123745792",
    ),
    "1nF 50V C0G": (
        "GRM1885C1H102JA01D",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588907605159268352",
    ),
    "22pF 50V C0G": (
        "CL10C220JB8NNNC",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8579707043277824000",
    ),
    "39pF 50V C0G": (
        "CL10C390JB81PNC",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8590224160770056192",
    ),
    "47pF 50V C0G": (
        "CL10C470JB8NNNC",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8579707064697999360",
    ),
    "680pF 50V C0G": (
        "CL10C681JB8NNNC",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8586178654925148160",
    ),
    "820pF 50V C0G": (
        "CGA3E2C0G1H821JT0Y0N",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588934303157243904",
    ),
    "1.5nF 50V C0G": (
        "CGA3E2C0G1H152JT0Y0N",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588933980334383104",
    ),
    "2.2nF 50V C0G": (
        "GRM1885C1H222JA01J",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8551517156874334208",
    ),
    "82nF 50V X7R": (
        "CL10B823KB8NNNC",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588948172350115840",
    ),
    "330nF 25V X7R": (
        "CC0603KRX7R8BB334",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8755242233409363968",
    ),
    "2.2uF 16V X5R": (
        "CL10A225KO8NNNC",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8579707132658442240",
    ),
    "2.2uF 25V X7R": (
        "GRM188Z71E225KE43D",
        "Capacitor_SMD:C_0603_1608Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588932123391119360",
    ),
    "4.7uF 50V X7R": (
        "GRM31CR71H475KA12L",
        "Capacitor_SMD:C_1206_3216Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588893061610393600",
    ),
    "10uF 25V X7R": (
        "GRM31CR71E106KA12L",
        "Capacitor_SMD:C_1206_3216Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588932779736240128",
    ),
    "10uF 25V X5R": (
        "CL21A106KAYNNNE",
        "Capacitor_SMD:C_0805_2012Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8579706853615591424",
    ),
    "22uF 16V X7R": (
        "EMK316BB7226ML-T",
        "Capacitor_SMD:C_1206_3216Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8590166435147653120",
    ),
    "47uF 6.3V X5R": (
        "CL31A476MQHNNNE",
        "Capacitor_SMD:C_1206_3216Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8586178714988662784",
    ),
    "47uF 25V X5R": (
        "HGC1210R5476M250NSVK",
        "Capacitor_SMD:C_1210_3225Metric",
        "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8590905388357402624",
    ),
}


def _catalog_cap(role: str, value: str, quantity: int = 1) -> Group:
    """Return one exact, physically sized capacitor from the reviewed catalog."""
    mpn, footprint, datasheet = _CATALOG_CAPACITORS[value]
    return _part(
        role,
        "C",
        f"{value}, {mpn}",
        "Device:C",
        footprint,
        mpn=mpn,
        datasheet=datasheet,
    ).model_copy(update={"quantity": quantity})


# These values are the highest voltage a *directly connected external control* may apply
# to the named EN pin.  They are not a declaration that every recipe's default enable
# implementation ties EN to VIN: TPS54331 has a 6 V EN absolute maximum and is instead
# enabled by its internal pull-up with a local shunt-to-ground disable jumper.
# AP6320x EN is specifically documented as a high-voltage input and may be tied to its
# 3.8--32 V VIN rail (DS41326 Rev. 3-2, pp. 4 and 10).
_ENABLE_CONTROL_MAX_V: dict[str, float] = {
    "tlv62569-3v3@3": 5.5,
    "ap63203-3v3@1": 32.0,
    "ap63205-5v@1": 32.0,
    "tps54331-adjustable@2": 6.0,
    "tps54331-12v-1a@2": 6.0,
    "lm2577t-adj-12v-boost@1": 18.0,
    "lm2577t-adj-2s-12v-boost@1": 18.0,
    "tps61088-2s-12v-1a@1": 18.0,
    "tps61088-host5-12v-1a@1": 18.0,
}


def enable_control_max_voltage(recipe: str) -> float | None:
    """Maximum verified voltage for an externally driven EN pin on this exact recipe."""
    return _ENABLE_CONTROL_MAX_V.get(recipe)


def _part(
    role: str,
    prefix: str,
    value: str,
    symbol: str,
    footprint: str,
    *,
    mpn: str | None = None,
    datasheet: str | None = None,
) -> Group:
    return Group(
        role=role,
        reference_prefix=prefix,
        value=value,
        symbol=symbol,
        footprint=footprint,
        sheet_role="power",
        mpn=mpn,
        datasheet=datasheet,
    )



def _passive(role: str, prefix: str, value: str, quantity: int = 1) -> Group:
    symbol, footprint = _C if prefix == "C" else _R if prefix == "R" else _L
    return _part(role, prefix, value, symbol, footprint).model_copy(update={"quantity": quantity})


def _source(
    url: str, title: str, *sections: str, reviewed_date: str = _DATE
) -> tuple[Source, ...]:
    return (
        Source(url=url, title=title, revision="current", reviewed_date=reviewed_date, sections=sections),
    )


def _power_recipe(
    *,
    recipe: str,
    family: str,
    exact_part: str | None,
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
    rated_output_current_a: float | None = None,
    source_documents: tuple[Source, ...] = (),
) -> RecipeDefinition:
    return RecipeDefinition(
        recipe=recipe,
        family=family,
        exact_part=exact_part,
        maturity="production",
        protected_aliases=tuple(
            value for value in (family, exact_part, *protected) if value
        ),
        identity_aliases=(exact_part,) if exact_part else (),
        rated_output_current_a=rated_output_current_a,
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
            f"{exact_part or family} reviewed application data",
            "Typical Application",
            "Electrical Characteristics",
            "Layout Guidelines",
        )
        + source_documents,
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
            "st-usblc6-2sc6:USBLC6-2SC6",
            "st-usblc6-2sc6:SOT-23-6_L2.9-W1.6-P0.95-LS2.8-BL",
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
        Pin(role="esd", pin="3", net="dp_conn"),
        Pin(role="esd", pin="2", net="gnd"),
        Pin(role="esd", pin="5", net="vbus"),
        Pin(role="esd", pin="6", net="dm_esd"),
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
    rated_output_current_a=1.0,
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
    rated_output_current_a: float,
    extra_nc: str | None = None,
    enable_pin: str | None = None,
    extra_assertions: tuple[Assertion, ...] = (),
) -> RecipeDefinition:
    no_connects = (NoConnect(role="regulator", pin=extra_nc),) if extra_nc else ()
    enable = (Pin(role="regulator", pin=enable_pin, net="enable"),) if enable_pin else ()
    return _power_recipe(
        recipe=recipe,
        family=family,
        exact_part=part,
        rated_output_current_a=rated_output_current_a,
        ports=(
            Port(name="input", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="output", direction="power"),
            *((Port(name="enable", direction="input", required=False, default_tie="input"),)
              if enable_pin else ()),
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
            *extra_assertions,
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
    rated_output_current_a=0.5,
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
    rated_output_current_a=0.25,
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
    rated_output_current_a=1.0,
    extra_assertions=(
        Assertion(
            code="ldo_thermal_derating",
            message="The 1 A rating holds only with an adequate copper/thermal area on the SOT-223 tab; derate for the assembled board",
        ),
    ),
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
        Port(name="feedback", direction="input", required=False, default_tie="output"),
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
        Pin(role="amplifier", pin="4", net="feedback"),
        Pin(role="decoupling", pin="1", net="vdd"),
        Pin(role="decoupling", pin="2", net="gnd"),
    ),
    source_url="https://ww1.microchip.com/downloads/aemDocuments/documents/MSLD/ProductDocuments/DataSheets/MCP6001-1R-1U-2-4-Low-Power-Op-Amp-DS20001733L.pdf",
    assertions=(
        Assertion(
            code="follower_feedback",
            message=(
                "Inverting input ties to the output for unity gain unless the design wires a "
                "feedback network"
            ),
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
    rated_output_current_a: float,
    inductor: str = "4.7uH",
    feedback_top: str = "100k",
    feedback_bottom: str = "22k",
    fixed_output: bool = False,
    catch_diode: Group | None = None,
    no_connect_pins: tuple[str, ...] = (),
) -> RecipeDefinition:
    parts = [
        _part("converter", "U", part, symbol, footprint, mpn=part),
        _passive("inductor", "L", inductor),
        _passive("input_cap", "C", "10uF", 2),
        _passive("output_cap", "C", "22uF", 2),
        _passive("feedback_top", "R", feedback_top),
        _passive("feedback_bottom", "R", feedback_bottom),
    ]
    if catch_diode is not None:
        parts.append(catch_diode)
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
    if catch_diode is not None:
        pins.extend(
            (
                Pin(role="catch_diode", pin="1", net="switch"),
                Pin(role="catch_diode", pin="2", net="gnd"),
            )
        )
    internal_nets = ["switch", "feedback"]
    if (pin := pins_map.get("en")) and pin not in no_connect_pins:
        pins.append(Pin(role="converter", pin=pin, net="enable"))
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
        rated_output_current_a=rated_output_current_a,
        ports=(
            Port(name="input", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="output", direction="power"),
            *(
                (Port(name="enable", direction="input", required=False, default_tie="input"),)
                if pins_map.get("en") and pins_map["en"] not in no_connect_pins
                else ()
            ),
        ),
        internal_nets=tuple(internal_nets),
        parameters={"output_voltage": output_voltage},
        allowed={"output_voltage": (output_voltage,)},
        parts=tuple(parts),
        pins=tuple(pins),
        no_connects=tuple(
            NoConnect(role="converter", pin=pin) for pin in no_connect_pins
        ),
        source_url=datasheet,
        assertions=(
            Assertion(
                code="buck_component_bounds",
                message="Inductor saturation, capacitor voltage, feedback tolerance, and thermal limits require independent review",
            ),
        ),
    )


TLV62569_3V3 = _buck(
    recipe="tlv62569-3v3@3",
    family="tlv62569-3v3",
    part="TLV62569DBVR",
    symbol="tlv62569:TLV62569DBVR",
    footprint="tlv62569:SOT-23-5_L3.0-W1.7-P0.95-LS2.8-BR",
    datasheet="https://www.ti.com/lit/ds/symlink/tlv62569.pdf",
    pins_map={"vin": "4", "gnd": "2", "sw": "3", "fb": "5", "en": "1"},
    output_voltage=3.3,
    rated_output_current_a=0.5,
    inductor="2.2uH",
)
TLV62569_3V3 = TLV62569_3V3.model_copy(update={
    "parts": tuple(
        _part("inductor", "L", "2.2uH FXL0630-2R2-M",
              "cjiang-fxl0630-2r2-m:FXL0630-2R2-M",
              "cjiang-fxl0630-2r2-m:FXL0630_Cjiang_Recommended", mpn="FXL0630-2R2-M",
              datasheet="https://datasheet.lcsc.com/datasheet/pdf/40dfb6cb4606b2ccaa4e89438a42f563.pdf?productCode=C167218").model_copy(
                  update={"sourcing_note": "LCSC C167218"})
        if part.role == "inductor" else
        _catalog_cap("input_cap", "10uF 25V X5R").model_copy(
            update={"sourcing_note": "LCSC C15850"})
        if part.role == "input_cap" else
        _catalog_cap("output_cap", "10uF 25V X7R", quantity=2).model_copy(
            update={"sourcing_note": "LCSC C77093"})
        if part.role == "output_cap" else
        part.model_copy(update={"sourcing_note": "LCSC C141836"})
        if part.role == "converter" else part
        for part in TLV62569_3V3.parts
    ),
    "pins": tuple(pin for pin in TLV62569_3V3.pins
                  if pin.role != "input_cap" or pin.index == 0),
    "placement_constraints": TLV62569_3V3.placement_constraints + (
        Placement(kind="owned_bypass", role="input_cap", parameters={
            "anchor_role": "converter", "power_net": "input", "ground_net": "gnd",
            "max_mm": 3.0,
        }),
    ),
    "source_documents": TLV62569_3V3.source_documents + (
        *_source("https://datasheet.lcsc.com/datasheet/pdf/40dfb6cb4606b2ccaa4e89438a42f563.pdf?productCode=C167218",
                 "Cjiang FXL0630", "Electrical characteristics", "Recommended land pattern"),
        *_source("https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588932779736240128",
                 "Murata GRM31CR71E106KA12L", "Electrical characteristics"),
        *_source("https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8579706853615591424",
                 "Samsung CL21A106KAYNNNE", "Part numbering", "Structure and dimensions", "Surface mounting"),
    ),
    "electrical_assertions": TLV62569_3V3.electrical_assertions + (
        Assertion(code="tlv62569_support_envelope", message=(
            "This composition is limited to 0.5 A, not the IC's 2 A rating. "
            "A single 10 uF/25 V X5R 0805 input ceramic is pin-owned within 3 mm "
            "of VIN/GND; TI recommends 4.7 uF nominal for most applications. "
            "Nominal input capacitance is not a guaranteed effective value. "
            "TI's table includes a 2.2 uH/22 uF nominal combination; these two "
            "10 uF capacitors total 20 uF nominal, and their effective capacitance "
            "under bias, temperature and tolerance and the actual loop stability "
            "require validation. The FXL0630 manufacturer lists saturation and "
            "heating test points as maximum/typical, not guaranteed minima."
        )),
        Assertion(code="tlv62569_operating_conditions", message=(
            "At 5.25 V input, 1.76 uH and the 1.5 MHz typical switching frequency, "
            "calculated ripple is approximately 0.464 A peak-to-peak at 3.3 V; "
            "this is illustrative, not a worst-case guarantee. Feedback reference "
            "and 1% divider tolerances imply approximately 3.208--3.450 V before "
            "ripple or transients. Verify actual EN bias, startup, layout, ripple, "
            "switching frequency, inductor saturation, dissipation and temperature "
            "on the finished board; none is established by these typical values."
        )),
    ),
})
AP63203_3V3 = _power_recipe(
    recipe="ap63203-3v3@1",
    family="ap63203-3v3",
    exact_part="AP63203WU-7",
    rated_output_current_a=1.0,
    parameters={"output_voltage": 3.3},
    allowed={"output_voltage": (3.3,)},
    ports=(
        Port(name="input", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="output", direction="power"),
        Port(name="enable", direction="input", required=False, default_tie="input"),
    ),
    internal_nets=("switch", "bootstrap"),
    parts=(
        _part(
            "converter", "U", "AP63203WU-7", "ap63203:AP63203WU-7",
            "ap63203:TSOT-26_L2.9-W1.6-P0.95-LS2.8-BL",
            mpn="AP63203WU-7",
            datasheet="https://www.diodes.com/assets/Datasheets/AP63200-AP63201-AP63203-AP63205.pdf",
        ),
        _part(
            "inductor", "L", "4.7uH FXL0630-4R7-M", "Device:L",
            "inductor-15uh-3a:IND-SMD_L7.0-W6.6_FXL0630",
            mpn="FXL0630-4R7-M",
            datasheet="https://xonstorage.z8.web.core.windows.net/pdf/changjiangmicroelectronics_fxl10501r0m_apr22_xonlink.pdf",
        ),
        _catalog_cap("input_cap", "4.7uF 50V X7R", quantity=8),
        _catalog_cap("input_hf_cap", "100nF 50V X7R"),
        _catalog_cap("output_cap", "10uF 25V X7R", quantity=4),
        _catalog_cap("bootstrap_cap", "100nF 50V X7R"),
    ),
    pins=(
        Pin(role="converter", pin="1", net="output"),
        Pin(role="converter", pin="2", net="enable"),
        Pin(role="converter", pin="3", net="input"),
        Pin(role="converter", pin="4", net="gnd"),
        Pin(role="converter", pin="5", net="switch"),
        Pin(role="converter", pin="6", net="bootstrap"),
        Pin(role="inductor", pin="1", net="switch"),
        Pin(role="inductor", pin="2", net="output"),
        *(Pin(role="input_cap", index=i, pin="1", net="input") for i in range(8)),
        *(Pin(role="input_cap", index=i, pin="2", net="gnd") for i in range(8)),
        Pin(role="input_hf_cap", pin="1", net="input"),
        Pin(role="input_hf_cap", pin="2", net="gnd"),
        *(Pin(role="output_cap", index=i, pin="1", net="output") for i in range(4)),
        *(Pin(role="output_cap", index=i, pin="2", net="gnd") for i in range(4)),
        Pin(role="bootstrap_cap", pin="1", net="bootstrap"),
        Pin(role="bootstrap_cap", pin="2", net="switch"),
    ),
    source_url="https://www.diodes.com/assets/Datasheets/AP63200-AP63201-AP63203-AP63205.pdf",
    source_documents=(
        *_source(
            "https://www.diodes.com/assets/Evaluation-Boards/AP63203-EVB-User-Guide.pdf",
            "AP63203 evaluation board", "External Components Selection", "BOM",
            reviewed_date="2026-11-03",
        ),
        *_source(
            "https://xonstorage.z8.web.core.windows.net/pdf/changjiangmicroelectronics_fxl10501r0m_apr22_xonlink.pdf",
            "Cjiang FXL inductor specification", "FXL0630 dimensions", "FXL0630-4R7-M electrical characteristics",
            reviewed_date="2026-11-03",
        ),
        *_source(
            "https://img.gme.cz/files/eshop_data/eshop_data/11/905-250/dsh.905-250.1.pdf",
            "Murata GRM31CR71H475KA12 characteristics", "DC bias characteristic, 25 C",
            reviewed_date="2026-11-03",
        ),
        *_source(
            "https://app.mectronic.com/dsheet/GRM31CR71E106KA12K_MURATA.pdf",
            "Murata GRM31CR71E106KA12 characteristics", "DC bias characteristic, 25 C",
            reviewed_date="2026-11-03",
        ),
    ),
    assertions=(
        Assertion(
            code="ap63203_support_bounds",
            message=(
                "The composition is limited to 1 A output and the IC's 3.8–32 V input "
                "operating range; input transients and reverse polarity require external protection. "
                "The 4.7 uH FXL0630 has 7.5 A saturation and 5.5 A heating ratings, "
                "33 milliohm maximum DCR. FB senses output directly; EN defaults to VIN, "
                "and 100 nF is connected between BST and SW."
            ),
        ),
        Assertion(
            code="ap63203_capacitance_and_thermal_review",
            message=(
                "At 25 C the manufacturer's typical DC-bias curves indicate approximately "
                "45% input capacitance at 32 V and 90% output capacitance at 3.3 V. "
                "Eight 4.7 uF input and four 10 uF output ceramics provide approximately "
                "13 uF and 28 uF respectively with additional 10% tolerance and 15% "
                "temperature allowances; these are estimates, not guaranteed bias minima. "
                "Place input capacitors at the VIN/GND loop and feedback at the output "
                "capacitors. Verify actual capacitance, stability, load transients, layout "
                "and thermal performance in the finished operating environment; no energized "
                "or thermal qualification is claimed."
            ),
        ),
    ),
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
    rated_output_current_a=2.0,
    fixed_output=True,
)
def _tps54331_buck(
    *,
    recipe: str,
    family: str,
    exact_part: str | None,
    output_voltage: float,
    rated_output_current_a: float,
    inductor: Group,
    input_cap: Group,
    output_cap: Group,
    feedback_top: str,
    feedback_bottom: str,
    comp_resistor: str,
    comp_capacitor: str,
    comp_hf_capacitor: str,
    source_documents: tuple[Source, ...],
    assertions: tuple[Assertion, ...],
) -> RecipeDefinition:
    """A complete TPS54331 buck with the data-sheet-required support circuit.

    EN is deliberately *not* tied to VIN: EN is limited to 6 V even though VIN
    may be 28 V. The local shunt is emitted only when the architecture explicitly
    requests ``enable_control=True``: installed it shorts EN to ground for OFF,
    and removed it leaves the documented internal pull-up to turn the part ON.
    An external binding is only valid for a 0--6 V or open-drain control and is
    protected by ``enable_control_max_voltage``.
    """
    datasheet = "https://www.ti.com/lit/ds/symlink/tps54331.pdf"
    return _power_recipe(
        recipe=recipe,
        family=family,
        exact_part=exact_part,
        rated_output_current_a=rated_output_current_a,
        parameters={"output_voltage": output_voltage, "enable_control": False},
        allowed={
            "output_voltage": (output_voltage,),
            "enable_control": (False, True),
        },
        ports=(
            Port(name="input", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="output", direction="power"),
            # This is an active-low, sink-to-ground control -- not a high-true
            # VIN jumper. ``enable_control=True`` adds the local shunt and
            # architecture must not bind a high-voltage rail to this port.
            Port(name="enable", direction="input", required=False),
        ),
        internal_nets=(
            "switch",
            "feedback",
            "bootstrap",
            "softstart",
            "compensation",
            "comp_rc",
            "enable",
        ),
        parts=(
            _part(
                "converter",
                "U",
                "TPS54331DDAR",
                "tps54331:TPS54331DDAR",
                "tps54331:SOIC-8_L4.9-W3.9-P1.27-LS6.0-BL-EP",
                mpn="TPS54331DDAR",
                datasheet=datasheet,
            ),
            inductor,
            input_cap,
            _catalog_cap("input_hf_cap", "10nF 50V X7R"),
            output_cap,
            _passive("feedback_top", "R", feedback_top),
            _passive("feedback_bottom", "R", feedback_bottom),
            _catalog_cap("bootstrap_cap", "100nF 16V X7R"),
            _catalog_cap("softstart_cap", "10nF 50V X7R"),
            _passive("comp_resistor", "R", comp_resistor),
            _catalog_cap("comp_capacitor", comp_capacitor),
            _catalog_cap("comp_hf_capacitor", comp_hf_capacitor),
            _part(
                "catch_diode",
                "D",
                "B340A-13-F 40V 3A Schottky",
                "Device:D_Schottky",
                "Diode_SMD:D_SMA",
                mpn="B340A-13-F",
                datasheet="https://www.diodes.com/assets/Datasheets/B340A.pdf",
            ),
            _part(
                "enable_jumper",
                "JP",
                "EN_DISABLE",
                "Connector_Generic:Conn_01x02",
                "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
            ),
        ),
        pins=(
            Pin(role="converter", pin="1", net="bootstrap"),
            Pin(role="converter", pin="2", net="input"),
            Pin(role="converter", pin="3", net="enable"),
            Pin(role="converter", pin="4", net="softstart"),
            Pin(role="converter", pin="5", net="feedback"),
            Pin(role="converter", pin="6", net="compensation"),
            Pin(role="converter", pin="7", net="gnd"),
            Pin(role="converter", pin="8", net="switch"),
            Pin(role="converter", pin="9", net="gnd"),
            Pin(role="inductor", pin="1", net="switch"),
            Pin(role="inductor", pin="2", net="output"),
            *(Pin(role="input_cap", index=index, pin="1", net="input") for index in range(2)),
            *(Pin(role="input_cap", index=index, pin="2", net="gnd") for index in range(2)),
            Pin(role="input_hf_cap", pin="1", net="input"),
            Pin(role="input_hf_cap", pin="2", net="gnd"),
            *(Pin(role="output_cap", index=index, pin="1", net="output") for index in range(2)),
            *(Pin(role="output_cap", index=index, pin="2", net="gnd") for index in range(2)),
            Pin(role="feedback_top", pin="1", net="output"),
            Pin(role="feedback_top", pin="2", net="feedback"),
            Pin(role="feedback_bottom", pin="1", net="feedback"),
            Pin(role="feedback_bottom", pin="2", net="gnd"),
            Pin(role="bootstrap_cap", pin="1", net="bootstrap"),
            Pin(role="bootstrap_cap", pin="2", net="switch"),
            Pin(role="softstart_cap", pin="1", net="softstart"),
            Pin(role="softstart_cap", pin="2", net="gnd"),
            Pin(role="comp_resistor", pin="1", net="compensation"),
            Pin(role="comp_resistor", pin="2", net="comp_rc"),
            Pin(role="comp_capacitor", pin="1", net="comp_rc"),
            Pin(role="comp_capacitor", pin="2", net="gnd"),
            Pin(role="comp_hf_capacitor", pin="1", net="compensation"),
            Pin(role="comp_hf_capacitor", pin="2", net="gnd"),
            Pin(role="catch_diode", pin="1", net="switch"),
            Pin(role="catch_diode", pin="2", net="gnd"),
            Pin(role="enable_jumper", pin="1", net="enable"),
            Pin(role="enable_jumper", pin="2", net="gnd"),
        ),
        source_url=datasheet,
        source_documents=source_documents,
        assertions=(
            Assertion(
                code="tps54331_enable_safe_states",
                message=(
                    "When enable_control=True, the local EN_DISABLE shunt installed "
                    "grounds EN and turns the converter off; removing it leaves "
                    "TPS54331's internal pull-up to enable the converter. Never "
                    "connect a 18 V or 24 V jumper directly to EN."
                ),
            ),
            Assertion(
                code="tps54331_power_loop",
                message=(
                    "Place the input ceramics, catch diode, inductor and output capacitors at "
                    "the IC power loop; solder the PowerPAD to GND with thermal vias and route "
                    "the feedback trace from the output-capacitor node away from PH."
                ),
            ),
            *assertions,
        ),
    )


TPS54331_ADJUSTABLE = _tps54331_buck(
    recipe="tps54331-adjustable@2",
    family="tps54331-adjustable",
    exact_part="TPS54331DDAR",
    output_voltage=3.3,
    rated_output_current_a=0.5,
    inductor=_part(
        "inductor", "L", "15uH FXL0630-150-M", "inductor-15uh-3a:FXL0630-150-M",
        "inductor-15uh-3a:IND-SMD_L7.0-W6.6_FXL0630", mpn="FXL0630-150-M",
        datasheet="https://datasheet.lcsc.com/datasheet/pdf/40dfb6cb4606b2ccaa4e89438a42f563.pdf?productCode=C177243",
    ),
    input_cap=_catalog_cap("input_cap", "4.7uF 50V X7R", quantity=2),
    output_cap=_catalog_cap("output_cap", "47uF 25V X5R", quantity=2),
    feedback_top="10.2k 1%",
    feedback_bottom="3.24k 1%",
    comp_resistor="29.4k 1%",
    comp_capacitor="1nF 50V C0G",
    comp_hf_capacitor="47pF 50V C0G",
    source_documents=(
        *_source("https://www.ti.com/lit/ds/symlink/tps54331.pdf", "TPS54331 datasheet", "Sections 6.1, 6.3, 6.5, 8.2.2.5--8.2.2.10 and 8.4"),
        *_source("https://www.ti.com/lit/ug/slvu247a/slvu247a.pdf", "TPS54331EVM-232", "Tables 1-1, 1-2 and 4-1"),
        *_source("https://datasheet.lcsc.com/datasheet/pdf/40dfb6cb4606b2ccaa4e89438a42f563.pdf?productCode=C177243", "Cjiang FXL series", "FXL0630 electrical table and dimensions"),
    ),
    assertions=(
        Assertion(code="tps54331_3v3_input_envelope", message=(
            "Nominal 24 V supply must remain within 7--28 V at the IC including normal tolerance and transients; 30 V is an absolute maximum, not an operating rating. "
            "An SMBJ26 clamp rated 42.1 V cannot guarantee IC protection. No industrial surge immunity is claimed. EN floats on to its internal pull-up or is grounded for off; never connect VIN directly to EN."
        )),
        Assertion(code="tps54331_3v3_current_and_setpoint", message=(
            "This composition is limited to 0.5 A continuous, not the IC's 3 A headline. "
            "With 28 V, 3.3 V, 12 uH (-20%) and 456 kHz the ideal inductor ripple is 0.532 A p-p and 0.5 A load peak is 0.766 A; current-dependent inductance, temperature and transients are not included. "
            "Cjiang lists 3.5 A saturation and 2.6 A heating as MAX columns, and 4 A and 3 A as typical; none is represented as a guaranteed minimum. "
            "10.2k/3.24k 1% and the specified 0.772--0.828 V reference yield approximately 3.154--3.487 V worst-case static feedback, inside ESP32-C3 3.0--3.6 V recommended supply; ripple and transients are additional and unmeasured."
        )),
        Assertion(code="tps54331_3v3_analytical_compensation", message=(
            "The 29.4k/1nF/47pF type-II compensation has approximately 5.41 kHz zero and 115 kHz high-frequency pole. "
            "TI's 54 uF effective-capacitance design example predicts roughly 25 kHz crossover; at 0.5 A and 54 uF its output pole is about 447 Hz and idealized phase margin about 65 degrees. "
            "This is an analytical starting point, not a guarantee: fitted capacitor DC bias, temperature, inductor, layout, loop margin and ESP load steps must be validated. "
            "TI EVM's reported 25 kHz/58 degrees is a measurement of its own fitted 6.8 uH network at 25 V/1 A, not of this 15 uH design."
        )),
        Assertion(code="tps54331_3v3_layout_validation", message=(
            "Place input ceramics and B340A catch diode close to VIN/PH/GND, bootstrap 100 nF between BOOT and PH, output bank at the inductor, and C3 module decoupling at its supply pins. "
            "Bond the PowerPAD to GND with thermal vias. Check output ripple, C3 RF transmit steps, startup, thermal rise and EMI on the actual board; none is measured here."
        )),
    ),
)


YLPTEC_K7803M_1000R3 = _power_recipe(
    recipe="ylptec-k7803m-1000r3@1",
    family="ylptec-k7803m-1000r3",
    exact_part="K7803M-1000R3",
    rated_output_current_a=0.5,
    parameters={"output_voltage": 3.3},
    allowed={"output_voltage": (3.3,)},
    ports=(
        Port(name="input", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="output", direction="power"),
    ),
    internal_nets=(),
    parts=(
        _part(
            "converter", "U", "K7803M-1000R3",
            "ylptec-k7803m-1000r3:K7803M-1000R3_C22749779",
            "ylptec-k7803m-1000r3:PWRM-TH_K78XXM-1000R3",
            mpn="K7803M-1000R3",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8603362912006254592",
        ),
        _part(
            "input_cap", "C", "10uF 50V X7R, CL31B106KBHNNNE",
            "Device:C", "Capacitor_SMD:C_1206_3216Metric",
            mpn="CL31B106KBHNNNE",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8586178918235164672",
        ),
        _catalog_cap("output_cap", "22uF 16V X7R"),
    ),
    pins=(
        Pin(role="converter", pin="1", net="input"),
        Pin(role="converter", pin="2", net="gnd"),
        Pin(role="converter", pin="3", net="output"),
        Pin(role="input_cap", pin="1", net="input"),
        Pin(role="input_cap", pin="2", net="gnd"),
        Pin(role="output_cap", pin="1", net="output"),
        Pin(role="output_cap", pin="2", net="gnd"),
    ),
    source_url="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8603362912006254592",
    assertions=(
        Assertion(code="k7803m_source_envelope", message=(
            "YLPTEC K7803M-1000R3 is a 3.3 V, 1 A rated integrated converter "
            "at 6--36 V input. This composition is restricted to 0.5 A and "
            "6 V <= VIN < 30 V; at 30 V or above the manufacturer requires an "
            "additional 33 uF low-ESR input electrolytic, not fitted here. "
            "Limit normal input and transient voltage to that envelope."
        )),
        Assertion(code="k7803m_support_and_thermal", message=(
            "Use the manufacturer's 10 uF/50 V input and 22 uF/10 V or higher "
            "voltage output ceramic support close to the module. Total output "
            "capacitance must not exceed 2200 uF. Natural-convection thermal "
            "derating is 100% to 61 C and approximately 60% at 85 C; this "
            "0.5 A design does not establish connector, copper, source or "
            "load-transient capability. No hot plug, input reverse polarity, "
            "parallel outputs, surge immunity or thermal margin is claimed."
        )),
    ),
)

# YLPTEC's design reference calls for input/output ceramics close to the
# converter. The 4.5 mm pad-center bound accommodates its SIP-3 courtyard and
# the two 1206 courtyards without relaxing either one.
YLPTEC_K7803M_1000R3 = YLPTEC_K7803M_1000R3.model_copy(update={
    "placement_constraints": YLPTEC_K7803M_1000R3.placement_constraints + tuple(
        Placement(kind="owned_bypass", role=role, parameters={
            "anchor_role": "converter", "power_net": net, "ground_net": "gnd",
            "max_mm": 4.5,
        })
        for role, net in (("input_cap", "input"), ("output_cap", "output"))
    ),
})

# This is a separate motor-only source, not an upgrade of a logic supply.  The
# converter's own output support remains on the raw side of the ideal diode.
K7803M_3V3_REVERSE_BLOCKED = YLPTEC_K7803M_1000R3.model_copy(update={
    "recipe": "k7803m-3v3-reverse-blocked@1",
    "family": "k7803m-3v3-reverse-blocked",
    "exact_part": None,
    "protected_aliases": ("k7803m-3v3-reverse-blocked",),
    "identity_aliases": (),
    "internal_nets": ("raw_output", "gate"),
    "parts": YLPTEC_K7803M_1000R3.parts + (
        _part("reverse_controller", "U", "LTC4412ES6#TRPBF",
              "Power_Management:LTC4412xS6", "Package_TO_SOT_SMD:TSOT-23-6",
              mpn="LTC4412ES6#TRPBF",
              datasheet="https://www.analog.com/media/en/technical-documentation/data-sheets/ltc4412.pdf").model_copy(update={"sourcing_note": "LCSC C514442"}),
        _part("reverse_fet", "Q", "AO3401A", "Transistor_FET:AO3401A",
              "Package_TO_SOT_SMD:SOT-23", mpn="AO3401A",
              datasheet="https://www.aosmd.com/sites/default/files/res/datasheets/AO3401A.pdf").model_copy(update={"sourcing_note": "LCSC C15127"}),
        _catalog_cap("reverse_decoupling", "4.7uF 50V X7R").model_copy(
            update={"sourcing_note": "LCSC C77096"}
        ),
    ),
    "pins": tuple(
        pin.model_copy(update={"net": "raw_output"})
        if (pin.role, pin.pin) in {("converter", "3"), ("output_cap", "1")} else pin
        for pin in YLPTEC_K7803M_1000R3.pins
    ) + (
        Pin(role="reverse_controller", pin="1", net="raw_output"),
        Pin(role="reverse_controller", pin="2", net="gnd"),
        Pin(role="reverse_controller", pin="3", net="gnd"),
        Pin(role="reverse_controller", pin="5", net="gate"),
        Pin(role="reverse_controller", pin="6", net="output"),
        Pin(role="reverse_fet", pin="1", net="gate"),
        Pin(role="reverse_fet", pin="2", net="output"),
        Pin(role="reverse_fet", pin="3", net="raw_output"),
        Pin(role="reverse_decoupling", pin="1", net="raw_output"),
        Pin(role="reverse_decoupling", pin="2", net="gnd"),
    ),
    "no_connects": (NoConnect(role="reverse_controller", pin="4"),),
    "placement_constraints": tuple(
        placement.model_copy(update={"parameters": {
            **placement.parameters, "power_net": "raw_output",
        }}) if placement.kind == "owned_bypass" and placement.role == "output_cap"
        else placement
        for placement in YLPTEC_K7803M_1000R3.placement_constraints
    ) + (Placement(kind="owned_bypass", role="reverse_decoupling", parameters={
        "anchor_role": "reverse_controller", "power_net": "raw_output",
        "ground_net": "gnd", "max_mm": 3.0,
    }),),
    "source_documents": YLPTEC_K7803M_1000R3.source_documents + (
        *_source("https://www.analog.com/media/en/technical-documentation/data-sheets/ltc4412.pdf",
                 "LTC4412", "Pin Functions", "Electrical Characteristics", "Applications Information"),
        *_source("https://www.aosmd.com/sites/default/files/res/datasheets/AO3401A.pdf",
                 "AOS AO3401A", "Electrical Characteristics"),
    ),
    "electrical_assertions": YLPTEC_K7803M_1000R3.electrical_assertions + (
        Assertion(code="motor_reverse_boundary", message=(
            "For the separate 3.3 V motor rail only, LTC4412 VIN senses the raw module "
            "output, SENSE the protected output, CTL is grounded and GATE drives the "
            "AOS AO3401A with drain at raw and source at the motor. Reverse turn-off "
            "is not instantaneous: the threshold is 10--32 mV and the 22 us maximum "
            "is specified for a 1 nF test load, not this board. Reverse leakage and "
            "transient reverse charge remain; module tolerance to that charge, effective "
            "raw capacitance, motor current and switching waveforms require independent "
            "verification. No arbitrary regenerative energy or surge guarantee is made."
        )),
        Assertion(code="motor_reverse_fet_envelope", message=(
            "Composition remains limited to 0.5 A, 6 V <= VIN < 30 V. At 25 C AOS "
            "specifies 85 milliohms maximum at -2.5 V gate drive; the controller "
            "regulates forward drop around 20 mV when possible. Check actual voltage, "
            "startup, temperature, SOA and thermal layout; this is not a stall-current "
            "or guaranteed dynamic motor-current rating."
        )),
    ),
})


TPS54331_12V_1A = _tps54331_buck(
    recipe="tps54331-12v-1a@2",
    family="tps54331-12v-1a",
    exact_part=None,
    output_voltage=12.0,
    rated_output_current_a=1.0,
    inductor=_part(
        "inductor",
        "L",
        "15uH 3A",
        "inductor-15uh-3a:FXL0630-150-M",
        "inductor-15uh-3a:IND-SMD_L7.0-W6.6_FXL0630",
        mpn="FXL0630-150-M",
        datasheet="https://lcsc.com/product-detail/Power-Inductors_15uH-20_C177243.html",
    ),
    input_cap=_catalog_cap("input_cap", "4.7uF 50V X7R", quantity=2),
    output_cap=_catalog_cap("output_cap", "47uF 25V X5R", quantity=2),
    feedback_top="140k 1%",
    feedback_bottom="10k 1%",
    comp_resistor="255k 1%",
    comp_capacitor="680pF 50V C0G",
    comp_hf_capacitor="22pF 50V C0G",
    source_documents=_source(
        "https://www.ti.com/lit/ds/symlink/tps54331.pdf",
        "TPS54331 18 V to 12 V, 1 A analytical design",
        "Sections 6.5, 8.2.2.5--8.2.2.10, 8.4",
    ),
    assertions=(
        Assertion(
            code="tps54331_12v_feedback_setpoint",
            message=(
                "The 140k/10k 1% divider gives 12.000 V from the nominal 0.800 V "
                "reference.  The data-sheet 0.772--0.828 V reference range gives "
                "11.58--12.42 V before resistor tolerance; observer evidence must "
                "calculate this from the connected divider rather than a net label."
            ),
        ),
        Assertion(
            code="tps54331_12v_1a_compensation",
            message=(
                "The 255k/680pF/22pF type-II network is a low-crossover "
                "analytical starting point from TPS54331 equations 20--28 for the "
                "fitted nominal 2x47uF output bank. MLCC DC-bias and temperature "
                "curves are not guaranteed minima for the fitted delivery MPN, so "
                "no effective-capacitance minimum is asserted; loop stability "
                "remains a validation requirement."
            ),
        ),
        Assertion(
            code="tps54331_12v_1a_current_bounds",
            message=(
                "At 18 V/12 V/1 A/570 kHz the 15uH inductor ripple is 0.468 A and "
                "peak is 1.234 A.  The 3 A inductor and B340A 40 V/3 A catch diode "
                "therefore exceed this continuous 1 A output path; do not select this "
                "recipe above its declared 1 A rating."
            ),
        ),
    ),
)


def expand_tps54331_buck(resolved):
    """Omit the EN shunt unless the owning requirement explicitly asks for it."""
    from .registry import expand_static_definition

    definitions = {
        TPS54331_ADJUSTABLE.recipe: TPS54331_ADJUSTABLE,
        TPS54331_12V_1A.recipe: TPS54331_12V_1A,
    }
    definition = definitions[resolved.selection.recipe]
    if resolved.parameters.get("enable_control", False):
        return expand_static_definition(definition, resolved)
    definition = definition.model_copy(
        update={
            "parts": tuple(
                group for group in definition.parts if group.role != "enable_jumper"
            ),
            "pins": tuple(
                pin for pin in definition.pins
                if pin.role != "enable_jumper"
                and not (
                    pin.role == "converter" and pin.pin == "3"
                    and "enable" not in resolved.selection.port_bindings
                )
            ),
            "no_connects": (
                *definition.no_connects,
                *((NoConnect(role="converter", pin="3"),)
                  if "enable" not in resolved.selection.port_bindings else ()),
            ),
        }
    )
    return expand_static_definition(definition, resolved)


def _tps61088_boost(
    *,
    recipe: str,
    family: str,
    exact_part: str | None,
    input_top: str,
    input_bottom: str,
    input_bulk: Group,
    source_contract: str,
    current_math: str,
) -> RecipeDefinition:
    """Build the bounded 12 V / 1 A TPS61088 boost implementation.

    The post-regulator eFuse is intentionally on the delivered output, rather
    than in the high-ripple input path.  It disconnects a connector short while
    the TPS61088 retains its datasheet-programmed cycle-by-cycle switch limit.
    """
    return _power_recipe(
        recipe=recipe,
        family=family,
        exact_part=exact_part,
        rated_output_current_a=1.0,
        parameters={"output_voltage": 12.0, "enable_control": False},
        allowed={"output_voltage": (12.0,), "enable_control": (False, True)},
        ports=(
            Port(name="input", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="output", direction="power"),
            # Active-low, local eFuse EN/UVLO sink; it is not a high-true VIN
            # input and must be realized by the architecture's sink control.
            Port(name="enable", direction="input", required=False),
        ),
        internal_nets=(
            "switch",
            "converter_enable",
            "vcc",
            "frequency",
            "bootstrap",
            "soft_start",
            "boost_output",
            "feedback",
            "compensation",
            "comp_rc",
            "ilim",
            "efuse_enable",
            "efuse_ovp",
            "efuse_ilim",
            "efuse_dvdt",
        ),
        parts=(
            _part(
                "converter",
                "U",
                "TPS61088RHLR",
                "tps61088:TPS61088RHLR",
                "tps61088:VQFN-20_L4.6-W3.6-P0.50-BL",
                mpn="TPS61088RHLR",
                datasheet="https://www.ti.com/lit/ds/symlink/tps61088.pdf",
            ),
            _part(
                "output_limiter",
                "U",
                "TPS25940LRVCR",
                "tps25940l:TPS25940LRVCR",
                "tps25940l:WQFN-20_L4.0-W3.0-P0.50-BL-EP",
                mpn="TPS25940LRVCR",
                datasheet="https://www.ti.com/lit/ds/symlink/tps25940.pdf",
            ),
            _part(
                "inductor",
                "L",
                "2.2uH XAL7070-222MEC, 19.6A typical Isat, 6.33mOhm max DCR",
                "xal7070-222mec:XAL7070-222MEC",
                "xal7070-222mec:XAL7070-222MEC",
                mpn="XAL7070-222MEC",
                datasheet="https://www.coilcraft.com/getmedia/1ba55433-bcc8-4838-9b21-382f497e12e0/xal7070.pdf",
            ),
            input_bulk,
            _catalog_cap("input_bypass", "22uF 16V X7R", quantity=2),
            _catalog_cap("vin_bypass", "100nF 16V X7R"),
            _catalog_cap("vcc_cap", "2.2uF 16V X5R"),
            _catalog_cap("bootstrap_cap", "100nF 16V X7R"),
            _catalog_cap("soft_start_cap", "47nF 50V X7R"),
            _part(
                "output_cap",
                "C",
                "22uF 25V X7R, CGA6P3X7R1E226MT000N",
                "Device:C",
                "Capacitor_SMD:C_1210_3225Metric",
                mpn="CGA6P3X7R1E226MT000N",
                datasheet="https://product.tdk.com/en/search/capacitor/ceramic/mlcc/info?part_no=CGA6P3X7R1E226M250AB",
            ).model_copy(update={"quantity": 3}),
            _passive("feedback_top_a", "R", "499k 1%"),
            _passive("feedback_top_b", "R", "3.16k 1%"),
            _passive("feedback_bottom", "R", "56k 1%"),
            _passive("comp_resistor", "R", "3.48k 1%"),
            _catalog_cap("comp_capacitor", "82nF 50V X7R"),
            _catalog_cap("comp_hf_capacitor", "39pF 50V C0G"),
            _passive("frequency_resistor", "R", "301k 1%"),
            _passive("switch_limit", "R", "49.9k 1%"),
            _passive("converter_enable_top", "R", input_top),
            _passive("converter_enable_bottom", "R", input_bottom),
            _passive("efuse_enable_top", "R", "75k 1%"),
            _passive("efuse_enable_bottom", "R", "10k 1%"),
            _passive("efuse_ovp_top", "R", "140k 1%"),
            _passive("efuse_ovp_bottom", "R", "10k 1%"),
            _passive("efuse_ilim_resistor", "R", "30.1k 1%"),
            _catalog_cap("efuse_dvdt_cap", "1.5nF 50V C0G"),
            _part(
                "enable_jumper",
                "JP",
                "EN_DISABLE",
                "Connector_Generic:Conn_01x02",
                "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
            ),
        ),
        pins=(
            Pin(role="converter", pin="1", net="vcc"),
            Pin(role="converter", pin="2", net="converter_enable"),
            Pin(role="converter", pin="3", net="frequency"),
            *(Pin(role="converter", pin=str(pin), net="switch") for pin in range(4, 8)),
            Pin(role="converter", pin="8", net="bootstrap"),
            Pin(role="converter", pin="9", net="input"),
            Pin(role="converter", pin="10", net="soft_start"),
            Pin(role="converter", pin="13", net="gnd"),
            *(Pin(role="converter", pin=str(pin), net="boost_output") for pin in range(14, 17)),
            Pin(role="converter", pin="17", net="feedback"),
            Pin(role="converter", pin="18", net="compensation"),
            Pin(role="converter", pin="19", net="ilim"),
            Pin(role="converter", pin="20", net="gnd"),
            Pin(role="converter", pin="21", net="gnd"),
            *(Pin(role="output_limiter", pin=str(pin), net="output") for pin in range(4, 9)),
            *(Pin(role="output_limiter", pin=str(pin), net="boost_output") for pin in range(9, 14)),
            Pin(role="output_limiter", pin="1", net="gnd"),
            Pin(role="output_limiter", pin="3", net="gnd"),
            Pin(role="output_limiter", pin="14", net="efuse_enable"),
            Pin(role="output_limiter", pin="15", net="efuse_ovp"),
            Pin(role="output_limiter", pin="16", net="gnd"),
            Pin(role="output_limiter", pin="17", net="efuse_ilim"),
            Pin(role="output_limiter", pin="18", net="efuse_dvdt"),
            Pin(role="output_limiter", pin="21", net="gnd"),
            Pin(role="inductor", pin="1", net="input"),
            Pin(role="inductor", pin="2", net="switch"),
            Pin(role="input_bulk", pin="1", net="input"),
            Pin(role="input_bulk", pin="2", net="gnd"),
            *(Pin(role="input_bypass", index=index, pin="1", net="input") for index in range(2)),
            *(Pin(role="input_bypass", index=index, pin="2", net="gnd") for index in range(2)),
            Pin(role="vin_bypass", pin="1", net="input"),
            Pin(role="vin_bypass", pin="2", net="gnd"),
            Pin(role="vcc_cap", pin="1", net="vcc"),
            Pin(role="vcc_cap", pin="2", net="gnd"),
            Pin(role="bootstrap_cap", pin="1", net="bootstrap"),
            Pin(role="bootstrap_cap", pin="2", net="switch"),
            Pin(role="soft_start_cap", pin="1", net="soft_start"),
            Pin(role="soft_start_cap", pin="2", net="gnd"),
            *(Pin(role="output_cap", index=index, pin="1", net="boost_output") for index in range(3)),
            *(Pin(role="output_cap", index=index, pin="2", net="gnd") for index in range(3)),
            Pin(role="feedback_top_a", pin="1", net="boost_output"),
            Pin(role="feedback_top_a", pin="2", net="feedback_top"),
            Pin(role="feedback_top_b", pin="1", net="feedback_top"),
            Pin(role="feedback_top_b", pin="2", net="feedback"),
            Pin(role="feedback_bottom", pin="1", net="feedback"),
            Pin(role="feedback_bottom", pin="2", net="gnd"),
            Pin(role="comp_resistor", pin="1", net="compensation"),
            Pin(role="comp_resistor", pin="2", net="comp_rc"),
            Pin(role="comp_capacitor", pin="1", net="comp_rc"),
            Pin(role="comp_capacitor", pin="2", net="gnd"),
            Pin(role="comp_hf_capacitor", pin="1", net="compensation"),
            Pin(role="comp_hf_capacitor", pin="2", net="gnd"),
            Pin(role="frequency_resistor", pin="1", net="frequency"),
            Pin(role="frequency_resistor", pin="2", net="switch"),
            Pin(role="switch_limit", pin="1", net="ilim"),
            Pin(role="switch_limit", pin="2", net="gnd"),
            Pin(role="converter_enable_top", pin="1", net="input"),
            Pin(role="converter_enable_top", pin="2", net="converter_enable"),
            Pin(role="converter_enable_bottom", pin="1", net="converter_enable"),
            Pin(role="converter_enable_bottom", pin="2", net="gnd"),
            Pin(role="efuse_enable_top", pin="1", net="boost_output"),
            Pin(role="efuse_enable_top", pin="2", net="efuse_enable"),
            Pin(role="efuse_enable_bottom", pin="1", net="efuse_enable"),
            Pin(role="efuse_enable_bottom", pin="2", net="gnd"),
            Pin(role="efuse_ovp_top", pin="1", net="boost_output"),
            Pin(role="efuse_ovp_top", pin="2", net="efuse_ovp"),
            Pin(role="efuse_ovp_bottom", pin="1", net="efuse_ovp"),
            Pin(role="efuse_ovp_bottom", pin="2", net="gnd"),
            Pin(role="efuse_ilim_resistor", pin="1", net="efuse_ilim"),
            Pin(role="efuse_ilim_resistor", pin="2", net="gnd"),
            Pin(role="efuse_dvdt_cap", pin="1", net="efuse_dvdt"),
            Pin(role="efuse_dvdt_cap", pin="2", net="gnd"),
            Pin(role="enable_jumper", pin="1", net="efuse_enable"),
            Pin(role="enable_jumper", pin="2", net="gnd"),
        ),
        no_connects=(
            NoConnect(role="converter", pin="11"),
            NoConnect(role="converter", pin="12"),
            NoConnect(role="output_limiter", pin="2"),
            NoConnect(role="output_limiter", pin="19"),
            NoConnect(role="output_limiter", pin="20"),
        ),
        source_url="https://www.ti.com/lit/ds/symlink/tps61088.pdf",
        source_documents=(
            *_source(
                "https://www.ti.com/lit/ds/symlink/tps25940.pdf",
                "TPS25940L latching output protection",
                "Sections 6, 7.5, 9.3.4, 10.2 and 11.1",
            ),
            *_source(
                "https://www.coilcraft.com/getmedia/1ba55433-bcc8-4838-9b21-382f497e12e0/xal7070.pdf",
                "Coilcraft XAL7070-222MEC",
                "Document 856-2 electrical data and Document 856-4 land pattern",
            ),
        ),
        assertions=(
            Assertion(
                code="tps61088_connected_synchronous_boost",
                message=(
                    "Pins 4--7 are the common SW node; the 2.2uH inductor runs from "
                    "actual input to SW and VOUT pins 14--16 feed the capacitors and "
                    "feedback divider. This is an integrated synchronous boost stage, "
                    "not a net-label or pass-through assertion."
                ),
            ),
            Assertion(
                code="tps61088_feedback_and_compensation",
                message=(
                    "The TPS61088 1.204V reference with 502.16k/56k divider computes "
                    "12.0004V nominal. The 3.48k/82nF/39pF compensation population is "
                    "a low-crossover analytical starting point and the 301k FSW-to-SW "
                    "resistor sets the documented 500kHz nominal point. The fitted "
                    "three-part 22uF bank has 66uF nominal capacitance, but no "
                    "delivery-MPN DC-bias/temperature minimum has been established. "
                    "No effective-capacitance minimum is asserted, and loop "
                    "stability remains a validation requirement."
                ),
            ),
            Assertion(
                code="tps61088_current_and_thermal_contract",
                message=(
                    f"{source_contract} {current_math} The 36.4C/W standard-board "
                    "thermal number is not a thermal license: require a four-layer "
                    "ground plane, exposed-pad thermal vias and <=85C ambient application "
                    "envelope; no higher output current is claimed."
                ),
            ),
            Assertion(
                code="tps61088_output_short_disconnect",
                message=(
                    "TPS25940LRVCR is downstream of the boost-output capacitors and "
                    "ahead of the delivered connector. Its 30.1k ILIM resistor gives "
                    "2.69--3.23A current-limit range and 5.22A worst fast-trip bound; "
                    "a connector short therefore cannot hold the converter output at "
                    "ground. All IN pins 9--13, OUT pins 4--8 and exposed pad 21 are "
                    "connected as required; the latching device resets only by taking "
                    "its EN/UVLO node low."
                ),
            ),
        ),
    )


TPS61088_2S_12V_1A = _tps61088_boost(
    recipe="tps61088-2s-12v-1a@1",
    family="tps61088-2s-12v-1a",
    exact_part="TPS61088RHLR",
    input_top="22k 1%",
    input_bottom="10k 1%",
    input_bulk=_part(
        "input_bulk",
        "C",
        "220uF 16V polymer, SPZ1CM221E08O00RAXXX",
        "Device:C_Polarized",
        "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm",
        mpn="SPZ1CM221E08O00RAXXX",
        datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588887533229125632",
    ),
    source_contract=(
        "A protected 2S source must provide >=3.0A continuous at 6.0V; input "
        "capacitors carry switching ripple and do not increase this source rating."
    ),
    current_math=(
        "At 6.0V, 12V/1A and conservative 75% efficiency, input average is "
        "2.667A. With 1.76uH minimum inductance, 200kHz datasheet-minimum "
        "frequency and D=0.5, ripple is 8.53A and peak is 6.93A. XAL7070-222MEC "
        "publishes 19.6A typical Isat and 13.2A at a 20C rise; those are component "
        "reference ratings, so board-temperature validation remains required."
    ),
)


TPS61088_HOST5_12V_1A = _tps61088_boost(
    recipe="tps61088-host5-12v-1a@1",
    family="tps61088-host5-12v-1a",
    exact_part=None,
    input_top="10k 1%",
    input_bottom="10k 1%",
    input_bulk=_part(
        "input_bulk",
        "C",
        "470uF 10V polymer, SPZ1AM471E08O00RAXXX",
        "Device:C_Polarized",
        "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm",
        mpn="SPZ1AM471E08O00RAXXX",
        datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588887533371731968",
    ),
    source_contract=(
        "Automatic source assumption: an external 5V host supply is rated >=3.4A "
        "continuous at 4.75V. A USB 2.0 port or unspecified 0.5A header does not "
        "meet this recipe's source contract."
    ),
    current_math=(
        "At 4.75V, 12V/1A and conservative 75% efficiency, input average is "
        "3.368A. With 1.76uH minimum inductance, 200kHz datasheet-minimum "
        "frequency and D=0.604, ripple is 8.17A and peak is 7.45A. "
        "XAL7070-222MEC publishes 19.6A typical Isat and 13.2A at a 20C rise; "
        "those are component reference ratings, so board-temperature validation "
        "remains required."
    ),
)


def expand_tps61088_boost(resolved):
    """Emit the latching eFuse's active-low disable shunt only when requested."""
    from .registry import expand_static_definition

    definition = {
        TPS61088_2S_12V_1A.recipe: TPS61088_2S_12V_1A,
        TPS61088_HOST5_12V_1A.recipe: TPS61088_HOST5_12V_1A,
    }[resolved.selection.recipe]
    if resolved.parameters.get("enable_control", False):
        return expand_static_definition(definition, resolved)
    return expand_static_definition(
        definition.model_copy(
            update={
                "parts": tuple(
                    part for part in definition.parts if part.role != "enable_jumper"
                ),
                "pins": tuple(
                    pin for pin in definition.pins if pin.role != "enable_jumper"
                ),
            }
        ),
        resolved,
    )


LM2577T_ADJ_12V_BOOST = _power_recipe(
    recipe="lm2577t-adj-12v-boost@1",
    family="lm2577t-adj-12v-boost",
    exact_part="LM2577T-ADJ/NOPB",
    rated_output_current_a=0.5,
    parameters={"output_voltage": 12.0, "enable_control": False},
    allowed={"output_voltage": (12.0,), "enable_control": (False, True)},
    ports=(
        Port(name="input", direction="power"),
        Port(name="gnd", direction="power"),
        Port(name="output", direction="power"),
        # The eFuse EN/UVLO node is pulled up locally; an external controller
        # may only sink it or drive it within the eFuse's 0--18 V range.
        Port(name="enable", direction="input", required=False),
    ),
    internal_nets=(
        "protected_input",
        "switch",
        "feedback",
        "feedback_top",
        "compensation",
        "comp_rc",
        "enable",
        "ovp",
        "ilim",
        "dvdt",
    ),
    parts=(
        _part(
            "current_limiter",
            "U",
            "TPS25940LRVCR",
            "tps25940l:TPS25940LRVCR",
            "tps25940l:WQFN-20_L4.0-W3.0-P0.50-BL-EP",
            mpn="TPS25940LRVCR",
            datasheet="https://www.ti.com/lit/ds/symlink/tps25940.pdf",
        ),
        _part(
            "converter",
            "U",
            "LM2577T-ADJ/NOPB",
            "lm2577t-adj-nopb:LM2577T-ADJ_NOPB",
            "lm2577t-adj-nopb:TO-220-5L_L10.2-W4.5-P1.70-BL",
            mpn="LM2577T-ADJ/NOPB",
            datasheet="https://www.ti.com/lit/ds/symlink/lm2577.pdf",
        ),
        _part(
            "input_bulk_cap",
            "C",
            "220uF 16V polymer, SPZ1CM221E08O00RAXXX",
            "Device:C_Polarized",
            "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm",
            mpn="SPZ1CM221E08O00RAXXX",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588887533229125632",
        ),
        _catalog_cap("efuse_input_bypass", "100nF 50V X7R"),
        _catalog_cap("protected_bypass", "100nF 50V X7R"),
        _part(
            "inductor",
            "L",
            "100uH Bourns SRP1770TA-101M, 5.3A Irms, 118mOhm max DCR",
            "srp1770ta-101m:SRP1770TA-101M",
            "srp1770ta-101m:SRP1770TA-101M",
            mpn="SRP1770TA-101M",
            datasheet="https://www.bourns.com/docs/product-datasheets/SRP1770TA.pdf",
        ),
        _part(
            "boost_diode",
            "D",
            "1N5821RLG 30V 3A Schottky",
            "Device:D_Schottky",
            "Diode_THT:D_DO-201AD_P15.24mm_Horizontal",
            mpn="1N5821RLG",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588887000547512320",
        ),
        _part(
            "output_cap",
            "C",
            "680uF 25V polymer, SPZ1EM681F14000RAXXX",
            "Device:C_Polarized",
            "Capacitor_THT:CP_Radial_D8.0mm_P3.50mm",
            mpn="SPZ1EM681F14000RAXXX",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8590224766033965057",
        ),
        _passive("feedback_top_a", "R", "48.7k 1%"),
        _passive("feedback_top_b", "R", "511R 1%"),
        _passive("feedback_bottom", "R", "5.62k 1%"),
        _passive("comp_resistor", "R", "2k 1%"),
        _catalog_cap("comp_capacitor", "330nF 25V X7R"),
        _passive("efuse_ilim", "R", "30.1k 1%"),
        _catalog_cap("efuse_dvdt", "1.5nF 50V C0G"),
        _passive("enable_top", "R", "35.7k 1%"),
        _passive("enable_bottom", "R", "10k 1%"),
        _passive("ovp_top", "R", "50.6k 1%"),
        _passive("ovp_bottom", "R", "10k 1%"),
        _part(
            "enable_jumper",
            "JP",
            "EN_DISABLE",
            "Connector_Generic:Conn_01x02",
            "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
        ),
    ),
    pins=(
        Pin(role="current_limiter", pin="1", net="gnd"),
        Pin(role="current_limiter", pin="3", net="gnd"),
        *(Pin(role="current_limiter", pin=str(pin), net="protected_input") for pin in range(4, 9)),
        *(Pin(role="current_limiter", pin=str(pin), net="input") for pin in range(9, 14)),
        Pin(role="current_limiter", pin="14", net="enable"),
        Pin(role="current_limiter", pin="15", net="ovp"),
        Pin(role="current_limiter", pin="16", net="gnd"),
        Pin(role="current_limiter", pin="17", net="ilim"),
        Pin(role="current_limiter", pin="18", net="dvdt"),
        Pin(role="current_limiter", pin="21", net="gnd"),
        Pin(role="converter", pin="1", net="compensation"),
        Pin(role="converter", pin="2", net="feedback"),
        Pin(role="converter", pin="3", net="gnd"),
        Pin(role="converter", pin="4", net="switch"),
        Pin(role="converter", pin="5", net="protected_input"),
        Pin(role="input_bulk_cap", pin="1", net="input"),
        Pin(role="input_bulk_cap", pin="2", net="gnd"),
        Pin(role="efuse_input_bypass", pin="1", net="input"),
        Pin(role="efuse_input_bypass", pin="2", net="gnd"),
        Pin(role="protected_bypass", pin="1", net="protected_input"),
        Pin(role="protected_bypass", pin="2", net="gnd"),
        Pin(role="inductor", pin="1", net="protected_input"),
        Pin(role="inductor", pin="2", net="switch"),
        Pin(role="boost_diode", pin="1", net="output"),
        Pin(role="boost_diode", pin="2", net="switch"),
        Pin(role="output_cap", pin="1", net="output"),
        Pin(role="output_cap", pin="2", net="gnd"),
        Pin(role="feedback_top_a", pin="1", net="output"),
        Pin(role="feedback_top_a", pin="2", net="feedback_top"),
        Pin(role="feedback_top_b", pin="1", net="feedback_top"),
        Pin(role="feedback_top_b", pin="2", net="feedback"),
        Pin(role="feedback_bottom", pin="1", net="feedback"),
        Pin(role="feedback_bottom", pin="2", net="gnd"),
        Pin(role="comp_resistor", pin="1", net="compensation"),
        Pin(role="comp_resistor", pin="2", net="comp_rc"),
        Pin(role="comp_capacitor", pin="1", net="comp_rc"),
        Pin(role="comp_capacitor", pin="2", net="gnd"),
        Pin(role="efuse_ilim", pin="1", net="ilim"),
        Pin(role="efuse_ilim", pin="2", net="gnd"),
        Pin(role="efuse_dvdt", pin="1", net="dvdt"),
        Pin(role="efuse_dvdt", pin="2", net="gnd"),
        Pin(role="enable_top", pin="1", net="input"),
        Pin(role="enable_top", pin="2", net="enable"),
        Pin(role="enable_bottom", pin="1", net="enable"),
        Pin(role="enable_bottom", pin="2", net="gnd"),
        Pin(role="ovp_top", pin="1", net="input"),
        Pin(role="ovp_top", pin="2", net="ovp"),
        Pin(role="ovp_bottom", pin="1", net="ovp"),
        Pin(role="ovp_bottom", pin="2", net="gnd"),
        Pin(role="enable_jumper", pin="1", net="enable"),
        Pin(role="enable_jumper", pin="2", net="gnd"),
    ),
    no_connects=(
        NoConnect(role="current_limiter", pin="2"),
        NoConnect(role="current_limiter", pin="19"),
        NoConnect(role="current_limiter", pin="20"),
    ),
    source_url="https://www.ti.com/lit/ds/symlink/lm2577.pdf",
    source_documents=(
        *_source(
            "https://www.ti.com/lit/ds/symlink/tps25940.pdf",
            "TPS25940L eFuse current-limit protection",
            "Sections 6, 7.5, 9.3.4, 10.2 and 11.1",
        ),
        *_source(
            "https://www.bourns.com/docs/product-datasheets/SRP1770TA.pdf",
            "Bourns SRP1770TA-101M",
            "Electrical data and recommended layout",
        ),
    ),
    assertions=(
        Assertion(
            code="lm2577_12v_connected_boost",
            message=(
                "VIN feeds the 100uH inductor, the inductor reaches LM2577 switch pin 4, "
                "and the 1N5821RLG is switch-anode/output-cathode; the 680uF output "
                "capacitor and complete 48.7k+511R/5.62k feedback divider are on that "
                "actual output. This is a boost regulator, not a labeled pass-through."
            ),
        ),
        Assertion(
            code="lm2577_feedback_setpoint",
            message=(
                "LM2577's 1.230 V nominal reference and the connected 49.211k/5.62k "
                "divider calculate 12.000378 V. The 1.206--1.254 V reference limits "
                "give 11.766--12.478 V before resistor tolerance; observers must report "
                "this divider-derived fact rather than reading the output net name."
            ),
        ),
        Assertion(
            code="lm2577_compensation_reference",
            message=(
                "The 2k/330nF compensation network, 100uH inductor, 1N5821RLG, "
                "220uF input bulk and 680uF output capacitor are the LM2577-ADJ "
                "5--10 V to 12 V/0.8 A manufacturer reference circuit."
            ),
        ),
        Assertion(
            code="lm2577_external_switch_fault_limit",
            message=(
                "TPS25940LRVCR is latching, with 30.1k ILIM: 2.69--3.23 A calculated "
                "ILIM range using the 89/R relation and ±8% limit accuracy; worst "
                "fast-trip is 1.5*3.23+0.375=5.22 A, below LM2577's mandatory 6 A "
                "external boost-switch limit. The 220uF bulk is upstream of the eFuse; "
                "only 100nF is at protected_input, so stored downstream energy cannot "
                "bypass that current limiter."
            ),
        ),
        Assertion(
            code="lm2577_500ma_power_path",
            message=(
                "At 4.75 V input, 12 V/0.5 A and conservative 75% efficiency, input "
                "average current is 1.684 A. With 80uH minimum inductance, 42 kHz "
                "minimum frequency, 0.9 V switch saturation and 0.5 V diode drop, "
                "maximum duty is 0.668, ripple is 0.944 A and peak is 2.156 A, below "
                "the 2.69 A minimum eFuse limit and the SRP1770TA-101M 5.3 A Irms "
                "reference rating. Its Irms rating is specified at 40 C rise; do not "
                "treat it as a board-temperature guarantee."
            ),
        ),
    ),
)


LM2577T_ADJ_2S_12V_BOOST = LM2577T_ADJ_12V_BOOST.model_copy(
    update={
        "recipe": "lm2577t-adj-2s-12v-boost@1",
        "family": "lm2577t-adj-2s-12v-boost",
        # The host-5V variant remains the sole LM2577 exact-part selector; this
        # separately configured use has the same BOM MPN on its converter group.
        "exact_part": None,
        "protected_aliases": ("lm2577t-adj-2s-12v-boost",),
        "identity_aliases": (),
        "parts": tuple(
            group.model_copy(
                update={
                    "value": {
                        "input_bulk_cap": "220uF 16V polymer, SPZ1CM221E08O00RAXXX",
                        "enable_top": "45.3k 1%",
                        "ovp_top": "91k 1%",
                    }.get(group.role, group.value)
                }
            )
            for group in LM2577T_ADJ_12V_BOOST.parts
        ),
        "electrical_assertions": (
            *LM2577T_ADJ_12V_BOOST.electrical_assertions,
            Assertion(
                code="lm2577_2s_input_envelope",
                message=(
                    "This variant binds only a protected 2S source of 6.0--8.4 V: "
                    "the 45.3k/10k UVLO divider enables above approximately 5.47 V "
                    "and the 91k/10k OVP divider trips at approximately 10.0 V. At "
                    "6.0 V/12 V/0.5 A/75% efficiency, average input current is "
                    "1.333 A; with 80uH/42 kHz/0.9 V switch/0.5 V diode, peak "
                    "current is 1.851 A, below the 2.69 A minimum eFuse limit."
                ),
            ),
        ),
    }
)


def expand_lm2577t_adj_12v_boost(resolved):
    """Emit the active-low enable shunt only for an explicitly requested control."""
    from .registry import expand_static_definition

    definition = {
        LM2577T_ADJ_12V_BOOST.recipe: LM2577T_ADJ_12V_BOOST,
        LM2577T_ADJ_2S_12V_BOOST.recipe: LM2577T_ADJ_2S_12V_BOOST,
    }[resolved.selection.recipe]
    if resolved.parameters.get("enable_control", False):
        return expand_static_definition(definition, resolved)
    definition = definition.model_copy(
        update={
            "parts": tuple(
                group for group in definition.parts if group.role != "enable_jumper"
            ),
            "pins": tuple(
                pin for pin in definition.pins if pin.role != "enable_jumper"
            ),
        }
    )
    return expand_static_definition(definition, resolved)




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
                        pin.model_copy(update={"net": {"6": "usb_dm", "4": "usb_dp"}[pin.pin]})
                        if pin.role == "esd" and pin.pin in {"6", "4"}
                        else pin
                    )
                    for pin in definition.pins
                    if pin.role != "usb_series"
                ),
            }
        )
    return expand_static_definition(definition, resolved)


# Reverse-polarity protection on the board input, built from a reviewed part the library already
# carries: the AONR21357 P-channel MOSFET is a production bundle with its own symbol, footprint
# and datasheet, and its own record warns to keep the gate inside the +-25 V VGS bound. The block
# is the canonical high-side arrangement -- drain (pins 5-8 and exposed pad 9) faces the raw
# input, source (1-3) faces the protected rail, the gate sits on ground through a series resistor
# -- so a reversed input holds the FET off and leaves only a reverse-biased body diode. Before
# this, a stated protection requirement had no family to select and no reviewed carrier to name:
# the architecture stage invented one, and the parts step could not resolve it.
REVERSE_POLARITY_PMOS = _power_recipe(
    recipe="reverse-polarity-pmos@1",
    family="reverse-polarity-pmos",
    exact_part="AONR21357",
    parts=(
        _part(
            "mosfet",
            "Q",
            "AONR21357",
            "aonr21357:AONR21357",
            "aonr21357:DFN-8_L3.0-W3.0-P0.65-BL",
            mpn="AONR21357",
        ),
        _passive("gate_resistor", "R", "10k"),
        _part(
            "gate_clamp",
            "D",
            "BZT52C10",
            "Device:D_Zener",
            "Diode_SMD:D_SOD-123",
            mpn="BZT52C10",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8756679576573890560",
        ),
    ),
    pins=(
        *(Pin(role="mosfet", pin=pin, net="input") for pin in ("5", "6", "7", "8", "9")),
        *(Pin(role="mosfet", pin=pin, net="output") for pin in ("1", "2", "3")),
        Pin(role="mosfet", pin="4", net="gate"),
        Pin(role="gate_resistor", pin="1", net="gate"),
        Pin(role="gate_resistor", pin="2", net="gnd"),
        Pin(role="gate_clamp", pin="1", net="output"),
        Pin(role="gate_clamp", pin="2", net="gate"),
    ),
    ports=(
        Port(name="input", direction="power"),
        Port(name="output", direction="power"),
        Port(name="gnd", direction="power"),
    ),
    internal_nets=("gate",),
    source_url=(
        "https://lcsc.com/product-detail/MOSFET_Alpha-Omega-Semicon-AOS-AONR21357_C431196.html"
    ),
    assertions=(
        Assertion(
            code="reverse_polarity_blocking",
            message=(
                "A reversed input reverse-biases the P-channel body diode and holds the FET off; "
                "BZT52C10 clamps source-to-gate voltage near 10 V, below the 25 V VGS limit"
            ),
        ),
    ),
)



TPS61175_BOOST_L7805_USB = _power_recipe(
    recipe="usb-5v-boost-linear@1",
    family="usb-5v-boost-linear",
    exact_part=None,
    rated_output_current_a=0.05,
    parameters={"output_voltage": 5.0},
    allowed={"output_voltage": (5.0,)},
    ports=(Port(name="input", direction="power"), Port(name="gnd", direction="power"),
           Port(name="output", direction="power")),
    internal_nets=("protected", "gate", "limited", "ilim", "switch", "boost_output",
                   "feedback", "frequency", "soft_start", "compensation", "comp_rc", "led_a"),
    parts=(
        _part("converter", "U", "TPS61175PWPR", "tps61175:TPS61175PWPR",
              "tps61175:HTSSOP-14_L5.0-W4.4-P0.65-LS6.4-BL-EP",
              mpn="TPS61175PWPR", datasheet="https://www.ti.com/lit/ds/symlink/tps61175.pdf"),
        _part("reverse_mosfet", "Q", "AONR21357", "aonr21357:AONR21357",
              "aonr21357:DFN-8_L3.0-W3.0-P0.65-BL", mpn="AONR21357"),
        _passive("gate_resistor", "R", "10k"),
        _part("gate_clamp", "D", "BZT52C10", "Device:D_Zener", "Diode_SMD:D_SOD-123",
              mpn="BZT52C10"),
        _part("input_limiter", "U", "TPS2553DBVR", "tps2553dbvr:TPS2553DBVR",
              "tps2553dbvr:SOT-23-6_DBV", mpn="TPS2553DBVR",
              datasheet="https://www.ti.com/lit/ds/symlink/tps2553.pdf"),
        _passive("ilim_resistor", "R", "61.9k 1%"),
        _catalog_cap("limiter_bypass", "100nF 50V X7R"),
        _part("inductor", "L", "15uH FXL0630-150-M", "inductor-15uh-3a:FXL0630-150-M",
              "inductor-15uh-3a:IND-SMD_L7.0-W6.6_FXL0630", mpn="FXL0630-150-M",
              datasheet="https://datasheet.lcsc.com/datasheet/pdf/40dfb6cb4606b2ccaa4e89438a42f563.pdf?productCode=C177243"),
        _catalog_cap("input_cap", "4.7uF 50V X7R", quantity=2),
        _catalog_cap("boost_output_cap", "4.7uF 50V X7R", quantity=2),
        _part("boost_output_bulk", "C", "22uF 35V polymer, 35SVPF22M",
              "Device:C_Polarized", "Capacitor_SMD:CP_Elec_6.3x5.9",
              mpn="35SVPF22M",
              datasheet="https://industrial.panasonic.com/cdbs/www-data/pdf/AAB8000/AAB8000C177.pdf"),
        _catalog_cap("soft_start_cap", "2.2uF 25V X7R"),
        _part("catch_diode", "D", "B360A FUXINSEMI 60V 3A Schottky", "fuxinsemi-b360a:B360A",
              "fuxinsemi-b360a:SMA_L4.3-W2.6-LS5.1-RD", mpn="B360A",
              datasheet="https://datasheet.lcsc.com/datasheet/pdf/4341698a2794575dc7c94af0a06dfb39.pdf?productCode=C6807778"),
        _part("feedback_top", "R", "86.6k 1%", *_R, mpn="0603WAF8662T5E"),
        _passive("feedback_bottom", "R", "10k 1%"),
        _part("frequency_resistor", "R", "80.6k 1%", *_R, mpn="0603WAF8062T5E"),
        _passive("comp_resistor", "R", "1k 1%"),
        _catalog_cap("comp_capacitor", "330nF 25V X7R"),
        _part("linear_regulator", "U", "L7805CV", "l7805cv:L7805CV",
              "l7805cv:TO-220-3_L10.0-W4.5-P2.54-L", mpn="L7805CV",
              datasheet="https://www.st.com/resource/en/datasheet/l7805cv.pdf"),
        _catalog_cap("linear_input_bypass", "330nF 25V X7R"),
        _catalog_cap("linear_output_bypass", "100nF 50V X7R"),
        _passive("preload_resistor", "R", "680 1%"),
        _passive("led_resistor", "R", "2.2k 1%"),
        _part("power_led", "D", "LTST-C190KGKT", "Device:LED",
              "LED_SMD:LED_0603_1608Metric", mpn="LTST-C190KGKT"),
    ),
    pins=(
        *(Pin(role="reverse_mosfet", pin=p, net="input") for p in ("5", "6", "7", "8", "9")),
        *(Pin(role="reverse_mosfet", pin=p, net="protected") for p in ("1", "2", "3")),
        Pin(role="reverse_mosfet", pin="4", net="gate"),
        Pin(role="gate_resistor", pin="1", net="gate"), Pin(role="gate_resistor", pin="2", net="gnd"),
        Pin(role="gate_clamp", pin="1", net="protected"), Pin(role="gate_clamp", pin="2", net="gate"),
        Pin(role="input_limiter", pin="1", net="protected"), Pin(role="input_limiter", pin="2", net="gnd"),
        Pin(role="input_limiter", pin="3", net="protected"), Pin(role="input_limiter", pin="5", net="ilim"),
        Pin(role="input_limiter", pin="6", net="limited"),
        Pin(role="ilim_resistor", pin="1", net="ilim"), Pin(role="ilim_resistor", pin="2", net="gnd"),
        Pin(role="limiter_bypass", pin="1", net="protected"), Pin(role="limiter_bypass", pin="2", net="gnd"),
        Pin(role="converter", pin="1", net="switch"), Pin(role="converter", pin="2", net="switch"),
        Pin(role="converter", pin="3", net="limited"), Pin(role="converter", pin="4", net="limited"),
        Pin(role="converter", pin="5", net="soft_start"), Pin(role="converter", pin="6", net="gnd"),
        Pin(role="converter", pin="7", net="gnd"), Pin(role="converter", pin="8", net="compensation"),
        Pin(role="converter", pin="9", net="feedback"), Pin(role="converter", pin="10", net="frequency"),
        *(Pin(role="converter", pin=p, net="gnd") for p in ("11", "12", "13", "14", "15")),
        Pin(role="inductor", pin="1", net="limited"), Pin(role="inductor", pin="2", net="switch"),
        Pin(role="catch_diode", pin="1", net="boost_output"), Pin(role="catch_diode", pin="2", net="switch"),
        *(Pin(role="input_cap", index=i, pin="1", net="limited") for i in range(2)),
        *(Pin(role="input_cap", index=i, pin="2", net="gnd") for i in range(2)),
        *(Pin(role="boost_output_cap", index=i, pin="1", net="boost_output") for i in range(2)),
        *(Pin(role="boost_output_cap", index=i, pin="2", net="gnd") for i in range(2)),
        Pin(role="boost_output_bulk", pin="1", net="boost_output"),
        Pin(role="boost_output_bulk", pin="2", net="gnd"),
        Pin(role="feedback_top", pin="1", net="boost_output"), Pin(role="feedback_top", pin="2", net="feedback"),
        Pin(role="feedback_bottom", pin="1", net="feedback"), Pin(role="feedback_bottom", pin="2", net="gnd"),
        Pin(role="frequency_resistor", pin="1", net="frequency"), Pin(role="frequency_resistor", pin="2", net="gnd"),
        Pin(role="soft_start_cap", pin="1", net="soft_start"), Pin(role="soft_start_cap", pin="2", net="gnd"),
        Pin(role="comp_resistor", pin="1", net="compensation"), Pin(role="comp_resistor", pin="2", net="comp_rc"),
        Pin(role="comp_capacitor", pin="1", net="comp_rc"), Pin(role="comp_capacitor", pin="2", net="gnd"),
        Pin(role="linear_regulator", pin="1", net="boost_output"), Pin(role="linear_regulator", pin="2", net="gnd"),
        Pin(role="linear_regulator", pin="3", net="output"),
        Pin(role="linear_input_bypass", pin="1", net="boost_output"), Pin(role="linear_input_bypass", pin="2", net="gnd"),
        Pin(role="linear_output_bypass", pin="1", net="output"), Pin(role="linear_output_bypass", pin="2", net="gnd"),
        Pin(role="preload_resistor", pin="1", net="output"), Pin(role="preload_resistor", pin="2", net="gnd"),
        Pin(role="led_resistor", pin="1", net="output"), Pin(role="led_resistor", pin="2", net="led_a"),
        Pin(role="power_led", pin="2", net="led_a"), Pin(role="power_led", pin="1", net="gnd"),
    ),
    no_connects=(NoConnect(role="input_limiter", pin="4"),),
    source_url="https://www.ti.com/lit/ds/symlink/tps61175.pdf",
    source_documents=(
        *_source("https://www.st.com/resource/en/datasheet/l7805cv.pdf",
                 "ST L7805CV linear regulator", "Tables 3 and 12", "Section 6.1"),
        *_source("https://industrial.panasonic.com/cdbs/www-data/pdf/AAB8000/AAB8000C177.pdf",
                 "Panasonic SVPF polymer capacitors", "Specifications", "Characteristics list"),
    ),
    assertions=(
        Assertion(code="usb_boost_linear_operating_envelope", message=(
            "USB source 4.75--5.5V feeds the reverse PMOS, then TPS2553 limiter, "
            "then a real TPS61175 boost to nominal 12V, then an L7805CV to 5V. "
            "The external aggregate load must not exceed 50mA; the 680 ohm preload "
            "and indicator are additional internal loads. ST guarantees 4.75--5.25V "
            "for 7--18V input and at least 5mA load at 0--125C junction; the preload "
            "provides at least 5mA in regulation. The 61.9k 1% limiter threshold "
            "is 378--480mA by TI's equations, not an output current rating."
        )),
        Assertion(code="usb_boost_linear_thermal_and_loop", message=(
            "At 12V nominal the 50mA external load, preload and indicator plus "
            "regulator ground current consume at most approximately 0.89W on the boost rail "
            "under the stated static assumptions. At assumed 60% efficiency and 4.6V "
            "after input protection this needs about 0.32A; "
            "efficiency is an assumption, not a guaranteed minimum. Free-air TO-220 "
            "50C/W is an ST thermal model, not a measurement; limit ambient to 0--50C "
            "and validate actual temperature and airflow. 86.6k/10k sets 11.872V "
            "nominal; reference and 1% divider limits give 11.424--12.333V before "
            "FB bias. 80.6k FREQ is near TI's 80k 1.2MHz typical table point, not "
            "its exact specified frequency test condition. 1k/330nF compensation is an "
            "analytical starting point, not a stability guarantee. The 22uF polymer "
            "is 17.6--26.4uF initial at 20C/120Hz; neither its operating-temperature "
            "and DC-bias capacitance nor the 4.7uF ceramic bank's effective values "
            "are bounded by those ratings. Actual loop, ripple, light-load DCM and "
            "pulse skipping, startup and transients require validation."
        )),
        Assertion(code="usb_boost_soft_start_limits", message=(
            "External 2.2uF on SS programs the TPS61175 COMP clamp, not a guaranteed "
            "output slew: the 6uA charge current is typical only. The boost diode also "
            "permits passive input-to-output charging before regulation. TPS2553 bounds "
            "its steady current-limit threshold; transient peaks and startup under load "
            "remain to be validated."
        )),
    ),
).model_copy(update={"maturity": "canary"})


def _tps552892_autonomous(
    *,
    recipe: str,
    family: str,
    exact_part: str | None,
    output_voltage: float,
    feedback_bottom: str,
    comp_resistor: str,
    comp_cap: str,
    comp_hf_cap: str,
    enable_top: str,
    enable_bottom: str,
    external_vcc: bool,
    source_contract: str,
    compensation_contract: str,
) -> RecipeDefinition:
    """Build a bounded TPS552892 buck-boost from TI's Figure 8-1 topology.

    The current-sense resistor is deliberately between the converter's VOUT/
    ISP node and the delivered rail.  FB therefore regulates the delivered
    rail and ISN sees the load side of the Kelvin shunt, as Figure 7-5 shows.
    """
    vcc_net = "input" if external_vcc else "vcc"
    mode_net = "input" if external_vcc else "vcc"
    external_vcc_pins = (
        (Pin(role="converter", pin="21", net="gnd"),)
        if external_vcc
        else ()
    )
    external_vcc_no_connects = (
        () if external_vcc else (NoConnect(role="converter", pin="21"),)
    )
    return _power_recipe(
        recipe=recipe,
        family=family,
        exact_part=exact_part,
        rated_output_current_a=2.0,
        parameters={"output_voltage": output_voltage},
        allowed={"output_voltage": (output_voltage,)},
        ports=(
            Port(name="input", direction="power"),
            Port(name="gnd", direction="power"),
            Port(name="output", direction="power"),
        ),
        internal_nets=(
            "enable",
            "frequency",
            "sw1",
            "sw2",
            "output_raw",
            "feedback",
            "compensation",
            "comp_rc",
            "boot1",
            "boot2",
            *(("vcc",) if not external_vcc else ()),
        ),
        parts=(
            _part(
                "converter",
                "U",
                "TPS552892RYQR",
                "tps552892:TPS552892RYQR",
                "tps552892:VQFN-21_L5.0-W3.0-P0.50-TL",
                mpn="TPS552892RYQR",
                datasheet="https://www.ti.com/lit/ds/symlink/tps552892.pdf",
            ),
            _part(
                "inductor",
                "L",
                "4.7uH XAL7070-472MEC, 15.2A typical Isat, 14.3mOhm max DCR",
                "xal7070:XAL7070-472MEC",
                "xal7070:XAL7070-472MEC",
                mpn="XAL7070-472MEC",
                datasheet="https://www.coilcraft.com/getmedia/1ba55433-bcc8-4838-9b21-382f497e12e0/xal7070.pdf",
            ),
            _part(
                "input_cap",
                "C",
                "10uF 50V X7R, C3225X7R1H106KT000E",
                "Device:C",
                "Capacitor_SMD:C_1210_3225Metric",
                mpn="C3225X7R1H106KT000E",
                datasheet="https://datasheet.lcsc.com/datasheet/pdf/03dd569e57786ab73740ef12cd5c3bff.pdf?productCode=C432929",
            ).model_copy(update={"quantity": 4}),
            _catalog_cap("input_hf_cap", "100nF 50V X7R"),
            _part(
                "output_cap",
                "C",
                "22uF 25V X7R, CGA6P3X7R1E226MT000N",
                "Device:C",
                "Capacitor_SMD:C_1210_3225Metric",
                mpn="CGA6P3X7R1E226MT000N",
                datasheet="https://product.tdk.com/en/search/capacitor/ceramic/mlcc/info?part_no=CGA6P3X7R1E226M250AB",
            ).model_copy(update={"quantity": 4}),
            _catalog_cap("output_hf_cap", "100nF 25V X7R"),
            _passive("feedback_top", "R", "100k 1%"),
            _passive("feedback_bottom", "R", feedback_bottom),
            _part(
                "sense_resistor",
                "R",
                "22mR 1%, 1W, WSL2512R0220FEA",
                "Device:R",
                "Resistor_SMD:R_2512_6332Metric",
                mpn="WSL2512R0220FEA",
                datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8589917278218592256",
            ),
            _catalog_cap("boot1_cap", "100nF 16V X7R"),
            _catalog_cap("boot2_cap", "100nF 16V X7R"),
            # TPS552892 pin 18 requires a ceramic >4.7uF. This 10uF/25V
            # exact MPN provides nominal margin; its biased effective value
            # is not asserted without delivery-part characterization.
            _catalog_cap("vcc_cap", "10uF 25V X7R"),
            _passive("frequency", "R", "49.9k 1%"),
            _passive("enable_top", "R", enable_top),
            _passive("enable_bottom", "R", enable_bottom),
            _passive("comp_resistor", "R", comp_resistor),
            _catalog_cap("comp_cap", comp_cap),
            _catalog_cap("comp_hf_cap", comp_hf_cap),
        ),
        pins=(
            Pin(role="converter", pin="1", net="enable"),
            Pin(role="converter", pin="2", net=mode_net),
            Pin(role="converter", pin="5", net="gnd"),
            Pin(role="converter", pin="6", net="frequency"),
            Pin(role="converter", pin="7", net="input"),
            Pin(role="converter", pin="8", net="sw1"),
            Pin(role="converter", pin="9", net="gnd"),
            Pin(role="converter", pin="10", net="sw2"),
            Pin(role="converter", pin="11", net="output_raw"),
            Pin(role="converter", pin="12", net="output_raw"),
            Pin(role="converter", pin="13", net="output"),
            Pin(role="converter", pin="14", net="feedback"),
            Pin(role="converter", pin="15", net="compensation"),
            Pin(role="converter", pin="17", net="gnd"),
            Pin(role="converter", pin="18", net=vcc_net),
            Pin(role="converter", pin="19", net="boot2"),
            Pin(role="converter", pin="20", net="boot1"),
            *external_vcc_pins,
            Pin(role="inductor", pin="1", net="sw1"),
            Pin(role="inductor", pin="2", net="sw2"),
            *(Pin(role="input_cap", index=i, pin="1", net="input") for i in range(4)),
            *(Pin(role="input_cap", index=i, pin="2", net="gnd") for i in range(4)),
            Pin(role="input_hf_cap", pin="1", net="input"),
            Pin(role="input_hf_cap", pin="2", net="gnd"),
            *(Pin(role="output_cap", index=i, pin="1", net="output_raw") for i in range(4)),
            *(Pin(role="output_cap", index=i, pin="2", net="gnd") for i in range(4)),
            Pin(role="output_hf_cap", pin="1", net="output_raw"),
            Pin(role="output_hf_cap", pin="2", net="gnd"),
            Pin(role="sense_resistor", pin="1", net="output_raw"),
            Pin(role="sense_resistor", pin="2", net="output"),
            Pin(role="feedback_top", pin="1", net="output"),
            Pin(role="feedback_top", pin="2", net="feedback"),
            Pin(role="feedback_bottom", pin="1", net="feedback"),
            Pin(role="feedback_bottom", pin="2", net="gnd"),
            Pin(role="boot1_cap", pin="1", net="boot1"),
            Pin(role="boot1_cap", pin="2", net="sw1"),
            Pin(role="boot2_cap", pin="1", net="boot2"),
            Pin(role="boot2_cap", pin="2", net="sw2"),
            Pin(role="vcc_cap", pin="1", net=vcc_net),
            Pin(role="vcc_cap", pin="2", net="gnd"),
            Pin(role="frequency", pin="1", net="frequency"),
            Pin(role="frequency", pin="2", net="gnd"),
            Pin(role="enable_top", pin="1", net="input"),
            Pin(role="enable_top", pin="2", net="enable"),
            Pin(role="enable_bottom", pin="1", net="enable"),
            Pin(role="enable_bottom", pin="2", net="gnd"),
            Pin(role="comp_resistor", pin="1", net="compensation"),
            Pin(role="comp_resistor", pin="2", net="comp_rc"),
            Pin(role="comp_cap", pin="1", net="comp_rc"),
            Pin(role="comp_cap", pin="2", net="gnd"),
            Pin(role="comp_hf_cap", pin="1", net="compensation"),
            Pin(role="comp_hf_cap", pin="2", net="gnd"),
        ),
        no_connects=(
            NoConnect(role="converter", pin="3"),
            NoConnect(role="converter", pin="4"),
            NoConnect(role="converter", pin="16"),
            *external_vcc_no_connects,
        ),
        source_url="https://www.ti.com/lit/ds/symlink/tps552892.pdf",
        source_documents=(
            *_source(
                "https://www.coilcraft.com/getmedia/1ba55433-bcc8-4838-9b21-382f497e12e0/xal7070.pdf",
                "Coilcraft XAL7070 high-current inductor",
                "XAL7070-472MEC electrical and mechanical data",
            ),
            *_source(
                "https://www.vishay.com/docs/30100/wsl.pdf",
                "Vishay WSL2512 current-sense resistor",
                "WSL2512 rating and resistance range",
            ),
            *_source(
                "https://product.tdk.com/en/search/capacitor/ceramic/mlcc/info?part_no=C3225X7R1H106K250AC",
                "TDK C3225X7R1H106K catalogue-family input MLCC characterization",
                "Reference DC-bias data, not a guaranteed minimum for the delivery code",
            ),
            *_source(
                "https://product.tdk.com/en/search/capacitor/ceramic/mlcc/info?part_no=CGA6P3X7R1E226M250AB",
                "TDK CGA6P3X7R1E226 family reference",
                "Family characterization, not a guaranteed DC-bias minimum for delivery code CGA6P3X7R1E226MT000N",
            ),
        ),
        assertions=(
            Assertion(
                code="tps552892_connected_four_switch_path",
                message=(
                    "The actual four-switch path is VIN(7)-SW1(8)-L1-SW2(10)-"
                    "VOUT(11). VOUT, ISP(12), all output capacitors and the high "
                    "side of the Kelvin shunt share output_raw; ISN(13), FB and the "
                    "delivered output share the far side. This is neither a "
                    "pass-through nor a feedback path that bypasses the shunt."
                ),
            ),
            Assertion(
                code="tps552892_bounded_current_and_source",
                message=(
                    f"{source_contract} The 22mOhm WSL2512 shunt has a 2.18--2.36A "
                    "current-loop threshold from the 48--52mV specification; the "
                    "2.0A continuous rating is below that protection threshold and "
                    "is not inferred from the IC's 7A minimum average-inductor "
                    "limit. At 2.0A it dissipates 88mW, within its 1W rating."
                ),
            ),
            Assertion(
                code="tps552892_compensation_and_support",
                message=(
                    f"{compensation_contract} FSW uses 49.9k for TI's 400kHz "
                    "reference operating point. BOOT1/SW1 and BOOT2/SW2 each have "
                    "the required 100nF capacitor; VCC has the required >4.7uF "
                    "ceramic represented by a 10uF nominal exact MPN. PG, CC and "
                    "CDC are intentionally open-drain/optional functions and are "
                    "explicit no-connects, not omitted pins."
                ),
            ),
        ),
    )


TPS552892_12V_2A = _tps552892_autonomous(
    recipe="tps552892-12v-2a@1",
    family="tps552892-autonomous-buck-boost",
    exact_part="TPS552892RYQR",
    output_voltage=12.0,
    feedback_bottom="11.1k 1%",
    comp_resistor="75R 1%",
    comp_cap="2.2uF 25V X7R",
    comp_hf_cap="2.2nF 50V C0G",
    enable_top="220k 1%",
    enable_bottom="39k 1%",
    external_vcc=False,
    source_contract=(
        "This is a 9--36V-input, 12V/2.0A output recipe. At the low 9V "
        "input, 12V/2A and a conservative 85% efficiency require 3.14A "
        "average source current; require a source rated >=3.2A continuous at "
        "9V. The 220k/39k EN divider switches on at approximately 8.17V and "
        "holds EN to 5.42V at 36V."
    ),
    compensation_contract=(
        "TPS552892 specifies effective CIN >=4.7uF and effective COUT from "
        "10uF to 1000uF (SLVSH28A, section 6.3). The fitted input bank is four "
        "C3225X7R1H106KT000E parts and the output bank is four "
        "CGA6P3X7R1E226MT000N parts, but their available family DC-bias curves "
        "are not guaranteed minima for these delivery codes. The 75ohm/2.2uF/"
        "2.2nF network is therefore only a nominal-component, 500Hz-crossover "
        "analytical starting point for the 9V/12V/2A boost corner; it does not "
        "establish a guaranteed effective capacitance or loop margin. A "
        "pad-bound observer must retain those data-sheet bounds and report the "
        "effective-capacitance predicate unverified until delivery-part evidence "
        "is added."
    ),
)


TPS552892_5V_2A = _tps552892_autonomous(
    recipe="tps552892-5v-2a@1",
    family="tps552892-autonomous-5v-buck-boost",
    exact_part=None,
    output_voltage=5.0,
    feedback_bottom="31.6k 1%",
    comp_resistor="221R 1%",
    comp_cap="330nF 25V X7R",
    comp_hf_cap="820pF 50V C0G",
    enable_top="27.4k 1%",
    enable_bottom="10k 1%",
    external_vcc=True,
    source_contract=(
        "This equal-input conversion is bounded to a regulated 4.75--5.25V "
        "source, such as a USB-C 5V/3A contract, and delivers 5V/2.0A. At "
        "4.75V and 85% efficiency, it requires 2.48A average input current; "
        "do not select it for an unspecified 0.5A USB source. VCC is supplied "
        "from that regulated input with EXTVCC grounded, as TI requires for "
        "an external 4.75--5.5V, >=100mA VCC source."
    ),
    compensation_contract=(
        "The equal-input 4.75V boost corner at 5V/2A has a calculated 212Hz "
        "RHP zero. The 221ohm/330nF/820pF type-II network targets a 25Hz "
        "crossover, below one fifth of that zero. It is a genuine four-switch "
        "buck-boost path through L1, never an input-to-output pass-through."
    ),
)


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
    YLPTEC_K7803M_1000R3,
    K7803M_3V3_REVERSE_BLOCKED,
    TPS54331_12V_1A,
    TPS552892_12V_2A,
    TPS552892_5V_2A,
    TPS61088_2S_12V_1A,
    TPS61088_HOST5_12V_1A,
    LM2577T_ADJ_2S_12V_BOOST,
    LM2577T_ADJ_12V_BOOST,
    REVERSE_POLARITY_PMOS,
    TPS61175_BOOST_L7805_USB,
)
