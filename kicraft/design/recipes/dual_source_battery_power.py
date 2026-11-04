"""Reviewed dual-source 5 V power path for host-plus-battery nodes.

The circuit keeps a host 5 V inlet physically distinct from a single-cell
Li-ion inlet.  A TPS61088 makes the battery branch into a regulated 5 V rail;
a TPS2113A then selects that rail or the host rail without reverse or
cross-conduction.  It is deliberately a bounded, low-current node supply, not
a charger or a generic battery-management block.
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
_DATE = "2026-10-11"


def _part(role: str, prefix: str, value: str, symbol: str, footprint: str, *, mpn: str | None = None, datasheet: str | None = None) -> Group:
    return Group(
        role=role, reference_prefix=prefix, value=value, symbol=symbol,
        footprint=footprint, mpn=mpn, datasheet=datasheet, sheet_role="power",
    )


DUAL_SOURCE_BATTERY_POWER = RecipeDefinition(
    recipe="dual-source-battery-power@1",
    family="dual-source-battery-power",
    exact_part="TPS2113APWR",
    maturity="production",
    protected_aliases=("dual-source-battery-power", "TPS2113APWR", "TPS2113A"),
    identity_aliases=("TPS2113APWR", "TPS2113A"),
    rated_output_current_a=0.5,
    required_sheet_roles=("power",),
    parameter_defaults={
        "host_voltage_v": 5.0,
        "host_voltage_min_v": 4.75,
        "host_voltage_max_v": 5.25,
        "battery_chemistry": "single-cell-liion",
        "battery_voltage_min_v": 3.0,
        "battery_voltage_max_v": 4.2,
        "load_current_a": 0.5,
        "include_host_connector": True,
    },
    allowed_parameters={
        "host_voltage_v": (5.0,), "host_voltage_min_v": (4.75,),
        "host_voltage_max_v": (5.25,), "battery_chemistry": ("single-cell-liion",),
        "battery_voltage_min_v": (3.0,), "battery_voltage_max_v": (4.2,),
        "load_current_a": (0.5,), "include_host_connector": (True, False),
    },
    ports=(
        Port(name="host_input", direction="power"),
        Port(name="battery_input", direction="power"),
        Port(name="output", direction="power"),
        Port(name="gnd", direction="power"),
    ),
    internal_nets=(
        "host_vsns", "boost_switch", "boost_vcc", "boost_frequency",
        "boost_bootstrap", "boost_soft_start", "boost_feedback", "boost_compensation",
        "boost_comp_rc", "boost_ilim", "battery_5v", "mux_enable", "mux_status",
        "mux_enable_high", "mux_current_limit",
    ),
    parts=(
        _part("host_connector", "J", "TSW-102-07-G-S HOST 5V", "Connector_Generic:Conn_01x02", "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical", mpn="TSW-102-07-G-S", datasheet="https://www.samtec.com/products/tsw-102-07-g-s"),
        _part("battery_connector", "J", "S2B-PH-SM4-TB(LF)(SN)", "jst-ph-2p:S2B-PH-SM4-TB", "jst-ph-2p:CONN-SMD_P2.00_S2B-PH-SM4-TB-LF-SN", mpn="S2B-PH-SM4-TB(LF)(SN)"),
        _part("battery_boost", "U", "TPS61088RHLR", "tps61088:TPS61088RHLR", "tps61088:VQFN-20_L4.6-W3.6-P0.50-BL", mpn="TPS61088RHLR", datasheet="https://www.ti.com/lit/ds/symlink/tps61088.pdf"),
        _part("power_mux", "U", "TPS2113APWR", "tps2113apwr:TPS2113APWR", "tps2113apwr:TSSOP-8_3x4p4mm_P0.65mm", mpn="TPS2113APWR", datasheet="https://www.ti.com/lit/ds/symlink/tps2113a.pdf"),
        _part("boost_inductor", "L", "2.2uH XAL7070-222MEC", "xal7070-222mec:XAL7070-222MEC", "xal7070-222mec:XAL7070-222MEC", mpn="XAL7070-222MEC"),
        _part("battery_input_cap", "C", "22uF 16V X7R", "Device:C", "Capacitor_SMD:C_1206_3216Metric", mpn="EMK316BB7226ML-T").model_copy(update={"quantity": 2}),
        _part("boost_input_bypass", "C", "100nF 16V X7R", *_C, mpn="CL10B104KO8NNNC"),
        _part("boost_vcc_cap", "C", "2.2uF 16V X5R", "Device:C", "Capacitor_SMD:C_0603_1608Metric", mpn="CL10A225KO8NNNC"),
        _part("boost_bootstrap_cap", "C", "100nF 16V X7R", *_C, mpn="CL10B104KO8NNNC"),
        _part("boost_soft_start_cap", "C", "47nF 50V X7R", *_C, mpn="CL10B473KB8NNNC"),
        _part("boost_output_cap", "C", "100uF 6.3V polymer tantalum, Panasonic 6TPE100MAZB", "Device:C_Polarized", "Capacitor_Tantalum_SMD:CP_EIA-3528-21_Kemet-B", mpn="6TPE100MAZB", datasheet="https://media.digikey.com/pdf/Data%20Sheets/Panasonic%20Capacitors%20PDFs/TPE_Series_POSCAP_Rev02_Oct_2017.pdf").model_copy(update={"quantity": 3}),
        _part("host_mux_bypass", "C", "100nF 16V X7R", *_C, mpn="CL10B104KO8NNNC"),
        _part("battery_mux_bypass", "C", "100nF 16V X7R", *_C, mpn="CL10B104KO8NNNC"),
        _part("output_mux_cap", "C", "22uF 16V X7R", "Device:C", "Capacitor_SMD:C_1206_3216Metric", mpn="EMK316BB7226ML-T"),
        _part("boost_feedback_top", "R", "178k 1%", *_R),
        _part("boost_feedback_bottom", "R", "56k 1%", *_R),
        _part("boost_comp_resistor", "R", "69.8k 1%, Yageo RC0603FR-0769K8L", *_R, mpn="RC0603FR-0769K8L"),
        _part("boost_comp_cap", "C", "15nF 50V C0G, TDK C2012C0G1H153JT000N", "Device:C", "Capacitor_SMD:C_0805_2012Metric", mpn="C2012C0G1H153JT000N"),
        _part("boost_comp_hf_cap", "C", "33pF 50V C0G, Samsung CL10C330JB8NNNC", *_C, mpn="CL10C330JB8NNNC"),
        _part("boost_frequency_resistor", "R", "301k 1%", *_R),
        _part("boost_limit_resistor", "R", "49.9k 1%", *_R),
        _part("host_priority_top", "R", "100k 1%", *_R),
        _part("host_priority_bottom", "R", "27k 1%", *_R),
        _part("mux_current_limit", "R", "400R 1%", *_R),
        _part("mux_enable_pulldown", "R", "100k", *_R),
        # The disable shunt must work with either source absent. These diodes
        # provide its high level without tying the host and battery branches.
        _part("host_enable_diode", "D", "SS14", "ss14:SS14", "ss14:SMA_L4.3-W2.6-LS5.0-RD", mpn="SS14"),
        _part("battery_enable_diode", "D", "SS14", "ss14:SS14", "ss14:SMA_L4.3-W2.6-LS5.0-RD", mpn="SS14"),
        _part("enable_jumper", "J", "TSW-102-07-G-S ENABLE DISABLE", "Connector_Generic:Conn_01x02", "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical", mpn="TSW-102-07-G-S", datasheet="https://www.samtec.com/products/tsw-102-07-g-s"),
    ),
    pins=(
        Pin(role="host_connector", pin="1", net="host_input"), Pin(role="host_connector", pin="2", net="gnd"),
        Pin(role="battery_connector", pin="1", net="battery_input"), Pin(role="battery_connector", pin="2", net="gnd"),
        Pin(role="battery_boost", pin="1", net="boost_vcc"), Pin(role="battery_boost", pin="2", net="battery_input"), Pin(role="battery_boost", pin="3", net="boost_frequency"),
        *(Pin(role="battery_boost", pin=str(pin), net="boost_switch") for pin in range(4, 8)),
        Pin(role="battery_boost", pin="8", net="boost_bootstrap"), Pin(role="battery_boost", pin="9", net="battery_input"), Pin(role="battery_boost", pin="10", net="boost_soft_start"), Pin(role="battery_boost", pin="13", net="gnd"),
        *(Pin(role="battery_boost", pin=str(pin), net="battery_5v") for pin in range(14, 17)),
        Pin(role="battery_boost", pin="17", net="boost_feedback"), Pin(role="battery_boost", pin="18", net="boost_compensation"), Pin(role="battery_boost", pin="19", net="boost_ilim"), Pin(role="battery_boost", pin="20", net="gnd"), Pin(role="battery_boost", pin="21", net="gnd"),
        Pin(role="power_mux", pin="1", net="mux_status"), Pin(role="power_mux", pin="2", net="mux_enable"), Pin(role="power_mux", pin="3", net="host_vsns"), Pin(role="power_mux", pin="4", net="mux_current_limit"), Pin(role="power_mux", pin="5", net="gnd"), Pin(role="power_mux", pin="6", net="battery_5v"), Pin(role="power_mux", pin="7", net="output"), Pin(role="power_mux", pin="8", net="host_input"),
        Pin(role="boost_inductor", pin="1", net="battery_input"), Pin(role="boost_inductor", pin="2", net="boost_switch"),
        *(Pin(role="battery_input_cap", index=index, pin="1", net="battery_input") for index in range(2)),
        *(Pin(role="battery_input_cap", index=index, pin="2", net="gnd") for index in range(2)),
        Pin(role="boost_input_bypass", pin="1", net="battery_input"), Pin(role="boost_input_bypass", pin="2", net="gnd"),
        Pin(role="boost_vcc_cap", pin="1", net="boost_vcc"), Pin(role="boost_vcc_cap", pin="2", net="gnd"),
        Pin(role="boost_bootstrap_cap", pin="1", net="boost_bootstrap"), Pin(role="boost_bootstrap_cap", pin="2", net="boost_switch"),
        Pin(role="boost_soft_start_cap", pin="1", net="boost_soft_start"), Pin(role="boost_soft_start_cap", pin="2", net="gnd"),
        *(Pin(role="boost_output_cap", index=index, pin="1", net="battery_5v") for index in range(3)),
        *(Pin(role="boost_output_cap", index=index, pin="2", net="gnd") for index in range(3)),
        Pin(role="host_mux_bypass", pin="1", net="host_input"), Pin(role="host_mux_bypass", pin="2", net="gnd"),
        Pin(role="battery_mux_bypass", pin="1", net="battery_5v"), Pin(role="battery_mux_bypass", pin="2", net="gnd"),
        Pin(role="output_mux_cap", pin="1", net="output"), Pin(role="output_mux_cap", pin="2", net="gnd"),
        Pin(role="boost_feedback_top", pin="1", net="battery_5v"), Pin(role="boost_feedback_top", pin="2", net="boost_feedback"), Pin(role="boost_feedback_bottom", pin="1", net="boost_feedback"), Pin(role="boost_feedback_bottom", pin="2", net="gnd"), Pin(role="boost_comp_resistor", pin="1", net="boost_compensation"), Pin(role="boost_comp_resistor", pin="2", net="boost_comp_rc"), Pin(role="boost_comp_cap", pin="1", net="boost_comp_rc"), Pin(role="boost_comp_cap", pin="2", net="gnd"), Pin(role="boost_comp_hf_cap", pin="1", net="boost_compensation"), Pin(role="boost_comp_hf_cap", pin="2", net="gnd"), Pin(role="boost_frequency_resistor", pin="1", net="boost_frequency"), Pin(role="boost_frequency_resistor", pin="2", net="boost_switch"), Pin(role="boost_limit_resistor", pin="1", net="boost_ilim"), Pin(role="boost_limit_resistor", pin="2", net="gnd"),
        Pin(role="host_priority_top", pin="1", net="host_input"), Pin(role="host_priority_top", pin="2", net="host_vsns"), Pin(role="host_priority_bottom", pin="1", net="host_vsns"), Pin(role="host_priority_bottom", pin="2", net="gnd"), Pin(role="mux_current_limit", pin="1", net="mux_current_limit"), Pin(role="mux_current_limit", pin="2", net="gnd"), Pin(role="mux_enable_pulldown", pin="1", net="mux_enable"), Pin(role="mux_enable_pulldown", pin="2", net="gnd"), Pin(role="enable_jumper", pin="1", net="mux_enable"), Pin(role="enable_jumper", pin="2", net="mux_enable_high"),
        Pin(role="host_enable_diode", pin="2", net="host_input"), Pin(role="host_enable_diode", pin="1", net="mux_enable_high"),
        Pin(role="battery_enable_diode", pin="2", net="battery_5v"), Pin(role="battery_enable_diode", pin="1", net="mux_enable_high"),
    ),
    no_connects=(
        NoConnect(role="battery_boost", pin="11"),
        NoConnect(role="battery_boost", pin="12"),
        NoConnect(role="battery_connector", pin="3"),
        NoConnect(role="battery_connector", pin="4"),
    ),
    placement_constraints=(
        Placement(kind="power_loop", role="battery_boost", parameters={"minimize_loop_area": True}),
        Placement(kind="power_loop", role="power_mux", parameters={"minimize_loop_area": True}),
        Placement(kind="decoupling_proximity", role="host_mux_bypass", parameters={"anchor_role": "power_mux", "max_mm": 3.0}),
        Placement(kind="decoupling_proximity", role="battery_mux_bypass", parameters={"anchor_role": "power_mux", "max_mm": 3.0}),
    ),
    electrical_assertions=(
        Assertion(code="separate_host_and_battery_inputs", message="Host header pin 1 and battery JST pin 1 are separate nets and reach the load only through distinct TPS2113A inputs."),
        Assertion(code="regulated_battery_5v_branch", message="TPS61088 has its documented battery-to-switch inductor path, three exact 100 uF polymer output capacitors, feedback divider, and retuned compensation; its regulated branch feeds TPS2113A IN2."),
        Assertion(code="tps2113a_control_and_isolation", message="TPS2113A EN is held low to enable automatic operation, the fitted shunt may drive EN high to disconnect OUT, VSNS gives host IN1 priority, and the device's reverse/cross-conduction blocking isolates sources."),
        Assertion(code="ttl_rail_voltage_margin", message="At 0.5 A, host path is 4.675 V minimum. TPS61088 VREF min/max plus 1% divider is 4.881 V to 5.188 V before TPS2113A; the battery load path is 4.806 V minimum and 5.188 V maximum, satisfying 4.5 V to 5.3 V."),
    ),
    source_documents=(
        Source(url="https://www.ti.com/lit/ds/symlink/tps2113a.pdf", title="TPS2113A auto-switching power mux", revision="C", reviewed_date=_DATE, sections=("Features", "Typical Application", "Auto-Switching Mode", "Reverse-Conduction Blocking", "Cross-Conduction Blocking")),
        Source(url="https://strawberry-linux.com/pub/tps61088.pdf", title="TPS61088 synchronous boost converter", revision="A", reviewed_date=_DATE, sections=("Recommended Operating Conditions", "Typical Application", "Layout Guidelines")),
        Source(url="https://media.digikey.com/pdf/Data%20Sheets/Panasonic%20Capacitors%20PDFs/TPE_Series_POSCAP_Rev02_Oct_2017.pdf", title="Panasonic TPE-B conductive polymer tantalum capacitors", revision="02", reviewed_date=_DATE, sections=("Characteristics list", "Specifications")),
        Source(url="https://www.lcsc.com/product-detail/C79109.html", title="LCSC C79109 6TPE100MAZB sourceability record", revision="observed", reviewed_date=_DATE, sections=("Manufacturer", "Model", "Stock")),
        Source(url="https://www.lcsc.com/product-detail/C445728.html", title="LCSC C445728 TDK 15 nF C0G compensation capacitor", revision="observed", reviewed_date=_DATE, sections=("Manufacturer", "Model", "Stock")),
        Source(url="https://www.lcsc.com/product-detail/C1663.html", title="LCSC C1663 Samsung 33 pF C0G compensation capacitor", revision="observed", reviewed_date=_DATE, sections=("Manufacturer", "Model", "Stock")),
        Source(url="https://www.lcsc.com/product-detail/C137689.html", title="LCSC C137689 Yageo 69.8 kohm compensation resistor", revision="observed", reviewed_date=_DATE, sections=("Manufacturer", "Model", "Stock")),
        Source(url="https://www.espressif.com/sites/default/files/documentation/esp32-c3-mini-1_datasheet_en.pdf", title="ESP32-C3-MINI-1 supply requirement", revision="current", reviewed_date=_DATE, sections=("Recommended Operating Conditions",)),
    ),
)


WAVE_G_DUAL_SOURCE_POWER_RECIPES = (DUAL_SOURCE_BATTERY_POWER,)


def expand_dual_source_battery_power(resolved):
    """Keep an existing host connector's ownership without emitting a duplicate."""
    from .registry import expand_static_definition

    definition = DUAL_SOURCE_BATTERY_POWER
    if not resolved.parameters["include_host_connector"]:
        definition = definition.model_copy(update={
            "parts": tuple(group for group in definition.parts if group.role != "host_connector"),
            "pins": tuple(pin for pin in definition.pins if pin.role != "host_connector"),
        })
    return expand_static_definition(definition, resolved)
