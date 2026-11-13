"""Filtered 3.3 V PWM to a bounded, bridge-driven passive speaker output."""
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


def _part(role, prefix, value, symbol, footprint, mpn=None):
    return Group(role=role, reference_prefix=prefix, value=value, symbol=symbol,
                 footprint=footprint, sheet_role="interface", mpn=mpn)


def _r(role, value):
    return _part(role, "R", value, "Device:R", "Resistor_SMD:R_0603_1608Metric")


def _c(role, value):
    return _part(role, "C", value, "Device:C", "Capacitor_SMD:C_0603_1608Metric")


PAM8302A_PWM_SPEAKER = RecipeDefinition(
    recipe="pam8302a-pwm-speaker@1",
    family="pwm-passive-speaker-driver",
    exact_part="PAM8302AADCR",
    default_for_family=True,
    external_interface_protocols=("passive-speaker-power-output",),
    maturity="production",
    protected_aliases=("pwm-passive-speaker-driver", "PAM8302AADCR"),
    identity_aliases=("PAM8302AADCR",),
    required_sheet_roles=("interface",),
    ports=(Port(name="vdd", direction="power"), Port(name="gnd", direction="power"),
           Port(name="pwm", direction="input")),
    # Conservative concurrent-load planning allocation, not a measured current.
    supply_current_reservations_a={"vdd": 0.6},
    internal_nets=("pwm_filtered", "audio_input", "audio_ref", "shutdown_delay",
                   "amp_out_p", "amp_out_n", "speaker_p", "speaker_n"),
    parts=(
        _part("amplifier", "U", "PAM8302AADCR", "pam8302a:PAM8302AADCR",
              "pam8302a:SOP-8_L4.9-W3.9-P1.27-LS6.0-BL", "PAM8302AADCR"),
        _part("speaker_header", "J", "PM254V-11-02-H85",
              "xfcn-pm254v-11-02-h85:PM254V-11-02-H85",
              "xfcn-pm254v-11-02-h85:XFCN_PM254V_11_02_H85", "PM254V-11-02-H85"),
        _r("pwm_series", "100k"), _r("pwm_shunt", "10k"),
        _c("pwm_filter", "3.3nF"), _c("input_coupling", "1uF"),
        _c("reference_coupling", "1uF"), _r("shutdown_pullup", "100k"),
        _c("shutdown_delay", "100nF"), _c("supply_decoupling", "1uF 10V"),
        _part("supply_bulk", "C", "10uF 25V X7R", "Device:C",
              "Capacitor_SMD:C_1206_3216Metric", "GRM31CR71E106KA12L"),
        _part("bead_positive", "FB", "BLM18PG221SN1D", "Device:FerriteBead",
              "Inductor_SMD:L_0603_1608Metric", "BLM18PG221SN1D"),
        _part("bead_negative", "FB", "BLM18PG221SN1D", "Device:FerriteBead",
              "Inductor_SMD:L_0603_1608Metric", "BLM18PG221SN1D"),
        _c("output_filter_positive", "1nF"), _c("output_filter_negative", "1nF"),
    ),
    pins=(
        Pin(role="amplifier", pin="1", net="shutdown_delay"),
        Pin(role="amplifier", pin="3", net="audio_input"),
        Pin(role="amplifier", pin="4", net="audio_ref"),
        Pin(role="amplifier", pin="5", net="amp_out_p"),
        Pin(role="amplifier", pin="6", net="vdd"),
        Pin(role="amplifier", pin="7", net="gnd"),
        Pin(role="amplifier", pin="8", net="amp_out_n"),
        Pin(role="speaker_header", pin="1", net="speaker_p"),
        Pin(role="speaker_header", pin="2", net="speaker_n"),
        Pin(role="pwm_series", pin="1", net="pwm"),
        Pin(role="pwm_series", pin="2", net="pwm_filtered"),
        Pin(role="pwm_shunt", pin="1", net="pwm_filtered"),
        Pin(role="pwm_shunt", pin="2", net="gnd"),
        Pin(role="pwm_filter", pin="1", net="pwm_filtered"),
        Pin(role="pwm_filter", pin="2", net="gnd"),
        Pin(role="input_coupling", pin="1", net="pwm_filtered"),
        Pin(role="input_coupling", pin="2", net="audio_input"),
        Pin(role="reference_coupling", pin="1", net="audio_ref"),
        Pin(role="reference_coupling", pin="2", net="gnd"),
        Pin(role="shutdown_pullup", pin="1", net="vdd"),
        Pin(role="shutdown_pullup", pin="2", net="shutdown_delay"),
        Pin(role="shutdown_delay", pin="1", net="shutdown_delay"),
        Pin(role="shutdown_delay", pin="2", net="gnd"),
        *(Pin(role=role, pin="1", net="vdd") for role in ("supply_decoupling", "supply_bulk")),
        *(Pin(role=role, pin="2", net="gnd") for role in ("supply_decoupling", "supply_bulk")),
        Pin(role="bead_positive", pin="1", net="amp_out_p"),
        Pin(role="bead_positive", pin="2", net="speaker_p"),
        Pin(role="bead_negative", pin="1", net="amp_out_n"),
        Pin(role="bead_negative", pin="2", net="speaker_n"),
        Pin(role="output_filter_positive", pin="1", net="speaker_p"),
        Pin(role="output_filter_positive", pin="2", net="gnd"),
        Pin(role="output_filter_negative", pin="1", net="speaker_n"),
        Pin(role="output_filter_negative", pin="2", net="gnd"),
    ),
    no_connects=(NoConnect(role="amplifier", pin="2"),),
    # The supply pads must be local to VDD; the GND return is proven against
    # actual connected copper, since both capacitor courts cannot fit there.
    placement_constraints=tuple(
        Placement(kind="owned_pin_local", role=role, parameters={
            "anchor_role": "amplifier", "owner_pin": "6", "member_pin": "1",
            "plane_owner_pin": "7", "plane_member_pin": "2",
            "plane_net": "gnd", "max_plane_via_mm": 3.0, "max_mm": 3.0,
        })
        for role in ("supply_decoupling", "supply_bulk")
    ),
    electrical_assertions=(
        Assertion(code="speaker_load", message="5 V +/-5%, 8 ohm passive speaker rated at least 2 W; reserve 0.6 A at 5 V. Output is bridge-tied: neither terminal is ground. Do not attach an externally grounded load."),
        Assertion(code="pwm_input", message="3.3 V PWM only; use at least 40 kHz carrier and center the audio waveform around 50% duty. Divider and 3.3 nF low-pass limit the amplifier input; 1 uF capacitors AC-couple both inputs."),
        Assertion(code="thermal", message="Estimate only, at low audio frequency: 1.65 V peak demodulated PWM into 100k/10k and an assumed 10k IC input gives 0.079 V peak, or about 1.18 V peak after 23.5 dB gain: 0.087 W and 0.104 A RMS into 8 ohms. At an assumed 25% efficiency plus 40 mW quiescent, dissipation is about 0.30 W; 115 C/W gives a 35 C rise, about 85 C junction at 50 C ambient. Efficiency and actual board thermal resistance are not guaranteed by this calculation: validate on hardware. Source reserve is not a thermal proof; shutdown is not a substitute for design."),
    ),
    source_documents=(
        Source(url="https://www.diodes.com/assets/Datasheets/PAM8302A.pdf", title="PAM8302A datasheet", revision="DS41333 Rev. 6-2", reviewed_date="2026-11-06", sections=("Typical Applications Circuit", "Electrical Characteristics", "Application Information")),
        Source(url="https://pim.murata.com/asset/pim4/ferriteBeadInductortypefilter/ENFA0003_PDF_FERRITEBEADINDUCTORTYPEFILTER?lastModifiedDatetime=20250707190934", title="BLM18 ferrite bead reference specification", revision="JENF243A", reviewed_date="2026-11-06", sections=("Ratings",)),
    ),
)
