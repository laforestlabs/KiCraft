"""USB-PD downstream series eFuse with hardware UVLO and controlled inrush.

TI SLVSE94G sections 5, 6.5, 8.3.1-8.3.7 and 9.2 are the design
basis. Ratings below are design allocations and calculations, not measurements.
The input must be a negotiated PD source, not an independently powered load.
"""
from __future__ import annotations

from .models import (
    RecipeDefinition,
    RecipeElectricalAssertion as Assertion,
    RecipeNoConnectSpec as NoConnect,
    RecipePinSpec as Pin,
    RecipePlacementConstraint as Placement,
    RecipePort as Port,
    RecipeSourceDocument as Source,
)
from .wave_b_power import _catalog_cap, _part, _passive

TI = "https://www.ti.com/lit/ds/symlink/tps2663.pdf"

TPS26631_PD_INRUSH_POWER_PATH = RecipeDefinition(
    recipe="tps26631-pd-inrush-power-path@1",
    family="tps26631-pd-power-path",
    exact_part="TPS26631RGER",
    default_for_family=True,
    maturity="production",
    pass_through=True,
    rated_output_current_a=1.8,
    supply_current_reservations_a={"input": 0.003},
    required_sheet_roles=("power",),
    parameter_defaults={
        "uvlo_rising_v_nominal": 7.477,
        "ovp_rising_v_nominal": 24.923,
        "current_limit_a_nominal": 2.0,
        "max_capacitive_load_uf": 250.0,
    },
    allowed_parameters={
        "uvlo_rising_v_nominal": (7.477,),
        "ovp_rising_v_nominal": (24.923,),
        "current_limit_a_nominal": (2.0,),
        "max_capacitive_load_uf": (250.0,),
    },
    ports=(
        Port(name="input", direction="power", supply_voltage_range_v=(8.0, 22.0)),
        Port(name="output", direction="power"),
        Port(name="power_good", direction="output", required=False),
        Port(name="gnd", direction="power"),
    ),
    internal_nets=("uvlo", "ovp", "ilim", "dvdt", "pgth"),
    parts=(
        _part("efuse", "U", "TPS26631RGER", "tps26631rger:TPS26631RGER",
              "tps26631rger:VQFN-24_L4.0-W4.0-P0.50-BL-EP2.5",
              mpn="TPS26631RGER", datasheet=TI),
        _passive("threshold_top", "R", "68k 1%"),
        _passive("threshold_middle", "R", "9.1k 1%"),
        _passive("threshold_bottom", "R", "3.9k 1%"),
        _passive("current_limit", "R", "9k 1%"),
        _passive("pgth_top", "R", "68k 1%"),
        _passive("pgth_bottom", "R", "13k 1%"),
        _passive("pg_pullup", "R", "33k 1%"),
        _passive("pg_pulldown", "R", "33k 1%"),
        _catalog_cap("slew_cap", "220nF 50V X7R"),
        _catalog_cap("input_bypass", "100nF 50V X7R"),
        _catalog_cap("output_bypass", "100nF 50V X7R"),
    ),
    pins=(
        *(Pin(role="efuse", pin=pin, net="input") for pin in ("1", "2", "5")),
        Pin(role="efuse", pin="6", net="uvlo"),
        Pin(role="efuse", pin="7", net="ovp"),
        *(Pin(role="efuse", pin=pin, net="gnd") for pin in ("8", "11", "25")),
        Pin(role="efuse", pin="9", net="dvdt"),
        Pin(role="efuse", pin="10", net="ilim"),
        Pin(role="efuse", pin="15", net="pgth"),
        Pin(role="efuse", pin="16", net="power_good"),
        *(Pin(role="efuse", pin=pin, net="output") for pin in ("17", "18")),
        Pin(role="threshold_top", pin="1", net="input"),
        Pin(role="threshold_top", pin="2", net="uvlo"),
        Pin(role="threshold_middle", pin="1", net="uvlo"),
        Pin(role="threshold_middle", pin="2", net="ovp"),
        Pin(role="threshold_bottom", pin="1", net="ovp"),
        Pin(role="threshold_bottom", pin="2", net="gnd"),
        Pin(role="current_limit", pin="1", net="ilim"),
        Pin(role="current_limit", pin="2", net="gnd"),
        Pin(role="pgth_top", pin="1", net="output"),
        Pin(role="pgth_top", pin="2", net="pgth"),
        Pin(role="pgth_bottom", pin="1", net="pgth"),
        Pin(role="pgth_bottom", pin="2", net="gnd"),
        Pin(role="pg_pullup", pin="1", net="output"),
        Pin(role="pg_pullup", pin="2", net="power_good"),
        Pin(role="pg_pulldown", pin="1", net="power_good"),
        Pin(role="pg_pulldown", pin="2", net="gnd"),
        Pin(role="slew_cap", pin="1", net="dvdt"),
        Pin(role="slew_cap", pin="2", net="gnd"),
        Pin(role="input_bypass", pin="1", net="input"),
        Pin(role="input_bypass", pin="2", net="gnd"),
        Pin(role="output_bypass", pin="1", net="output"),
        Pin(role="output_bypass", pin="2", net="gnd"),
    ),
    # SHDN has an internal 2.7 V pullup (section 6.5); float to enable.
    # B_GATE/DRV are left open only because no external blocking FET is used.
    no_connects=tuple(NoConnect(role="efuse", pin=pin) for pin in
                      ("3", "4", "12", "13", "14", "19", "20", "21", "22", "23", "24")),
    placement_constraints=(
        Placement(kind="power_loop", role="efuse", parameters={"minimize_loop_area": True}),
        Placement(kind="decoupling_proximity", role="efuse", parameters={"max_mm": 3.0}),
        # RGE pins 1/2 are the FET input; pin 5 is IN_SYS, not the
        # main current input. Pin 17 is one of the two FET outputs.
        Placement(kind="owned_bypass", role="input_bypass", parameters={
            "anchor_role": "efuse", "anchor_power_pin": "2", "power_net": "input",
            "ground_net": "gnd", "max_mm": 3.0,
        }),
        Placement(kind="owned_bypass", role="output_bypass", parameters={
            "anchor_role": "efuse", "anchor_power_pin": "17", "power_net": "output",
            "ground_net": "gnd", "max_mm": 3.0,
        }),
        Placement(kind="thermal_current_path", role="efuse",
                  parameters={"independent_review_required": True, "copper_area_required": True,
                              "outer_copper_oz": 2.0, "dual_layer_ground": True,
                              "local_radius_mm": 3.0, "minimum_local_ground_area_mm2": 4.0}),
    ),
    electrical_assertions=(
        Assertion(code="pd_hardware_uvlo_ovp", message=(
            "68k/9.1k/3.9k 1% ladder: UVLO rising nominal 7.477V and OVP rising nominal "
            "24.923V, from 1.2V threshold; comparator/leakage and resistor tolerances "
            "must keep 5V off and permit 8–22V. 5V pre-PD is not connected to bulk. "
            "Use only a compliant negotiated PD source within 22V; the nominal "
            "24.9V OVP is not a qualified clamp for arbitrary 25V-rail surges.")),
        Assertion(code="pd_limited_inrush", message=(
            "PGTH senses OUT through 68k/13k and PGOOD clamps power_good low until "
            "the FET gate is enhanced. The 33k/33k output divider drives both "
            "AP6320x EN pins only after the cold-start ramp; at 9–22V it is nominally "
            "4.5–11V, below their 32V EN rating. 220nF dVdT yields nominal 41ms "
            "at 9V (TI eq. 2). Assuming >=70% effective capacitance, worst charging "
            "current and gain give >=23ms at 9V and <=95mA for 250uF. Recovery with "
            "PGTH still high uses fast restart, so cold-start slew is not a fault guarantee.")),
        Assertion(code="pd_current_and_thermal", message=(
            "9k ILIM gives 2A nominal; TI specifies 1.84–2.16A at 9k. Scaling "
            "the endpoints using the 18/R equation for a 1% resistor gives "
            "approximately 1.822–2.182A. 1.8A continuous is a board design "
            "allocation, not a 60V rating: 53mOhm worst "
            "case yields 0.212W at 2A, requiring soldered EP, adequate ground copper, "
            "trace sizing, and board-level thermal qualification. For 250uF at 22V "
            "capacitor charging alone dissipates 0.0605J and <=2.09W peak; regulator "
            "EN is held off during cold ramp. After PGOOD, TPS26631 permits 2x current "
            "for up to 31ms: at 4.364A this is 0.591 A²s, plus fast-trip transient. "
            "Source, connector, copper and thermal behavior require qualification.")),
        Assertion(code="pd_reverse_path_and_raw_cap", message=(
            "No external blocking FET: no reverse-current or reverse-polarity protection. "
            "Never connect another source/backfeed to output. Local raw bypass is "
            "100nF +10%; include all other raw-side capacitances and tolerances in "
            "USB 10uF limit. FLT/IMON unused, MODE grounded for auto retry.")),
    ),
    source_documents=(
        Source(
            url=TI, title="Texas Instruments TPS2663x eFuse datasheet",
            revision="SLVSE94G, June 2024", reviewed_date="2026-11-06",
            sections=("5 pin functions", "6.5 electrical characteristics",
                "8.3.1 inrush", "8.3.2 PGTH", "8.3.3 UVLO", "8.3.4 OVP",
                "8.3.5 reverse blocking", "8.3.7 overcurrent", "9.2 design equations", "9.6 layout")),
        Source(
            url="https://www.diodes.com/assets/Datasheets/AP63200-AP63201-AP63203-AP63205.pdf",
            title="Diodes Incorporated AP6320x regulator datasheet",
            revision="DS41326 Rev. 3-2", reviewed_date="2026-11-06",
            sections=("pin descriptions", "electrical characteristics", "3 Enable")),
    ),
)
