"""Versioned, evaluation-only fulfillment contracts for the frozen benchmark.

This module deliberately does not participate in generation.  It makes the original
brief obligations independently checkable from artifacts and keeps approved corpus
revisions separate from the immutable production brief list.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS, ORIGINAL_BENCHMARK_CORPUS_VERSION

ORIGINAL_CORPUS_VERSION = ORIGINAL_BENCHMARK_CORPUS_VERSION

APPROVED_5V_DEVICE_CORPUS_VERSION = "benchmark-approved-5v-device-v1"
CONTRACT_SCHEMA_VERSION = 1

# All evidence claimed as positive must point at a real artifact under the run.
ARTIFACT_EVIDENCE_KINDS = frozenset({"artifact", "measurement", "tool-report", "manufacturer-source"})
RESULT_STATUSES = frozenset({"pass", "fail", "unverified"})
FEASIBILITY_STATUSES = frozenset({"reviewed_feasible", "specification_conflict", "not_yet_reviewed"})

COMMON_OBLIGATIONS = (
    ("sourceable-parts", "Real, sourceable and appropriately rated parts are evidenced.", "part_inventory", {}),
    ("pin-footprint-mapping", "Symbol, footprint, package, and pin mappings are evidenced.", "pin_mapping", {}),
    ("complete-required-connections", "Every required electrical connection is evidenced by the delivered board and wiring graph.", "gate", {"gate": "complete_required_connections"}),
    ("exported-artifacts", "Faithful exported fabrication artifacts are present and tied to this run.", "artifacts", {}),
    ("erc", "Applicable ERC gate passes with its report artifact.", "gate", {"gate": "erc"}),
    ("drc", "Applicable DRC/manufacturing gate passes with its report artifact.", "gate", {"gate": "drc"}),
    ("programming-when-applicable", "Where programmable hardware is present, its programming gate passes with report evidence.", "applicable_gate", {"gate": "programming"}),
    ("geometry-when-applicable", "Where the brief requires delivered geometry, its geometry gate passes with report evidence.", "applicable_gate", {"gate": "geometry"}),
)

# Each tuple is (stable obligation id suffix, independently checked obligation,
# evaluator, evaluator parameters).  These are reviewed against the *original*
# brief, not reconstructed from a generated design state.
_SPECIFIC: dict[str, tuple[tuple[str, str, str, dict[str, Any]], ...]] = {
    "rc-lowpass-bnc": (
        ("bnc-count", "Two physical BNC connectors are fitted.", "part_class_count", {"part_class": "bnc_connector", "minimum": 2}),
        ("trim-pot", "A trim potentiometer is in the passive RC signal path.", "part_class_count", {"part_class": "trim_potentiometer", "minimum": 1}),
        ("adjustable-response", "The measured/calculated cutoff response covers the declared adjustment range.", "numeric_range", {"fact": "cutoff_hz"}),
        ("no-mcu", "No microcontroller is fitted.", "part_class_exact", {"part_class": "microcontroller", "value": 0}),
    ),
    "r2r-dac": (
        ("logic-inputs", "Eight distinct logic inputs terminate on a header.", "channel_count", {"channel": "logic_input", "minimum": 8}),
        ("r2r-network", "The resistor ladder is an electrically evidenced R-2R network.", "net_paths", {"paths": ["r2r_ladder"]}),
        ("buffered-output", "An op-amp buffers the analog output within recorded supply/load limits.", "net_paths", {"paths": ["r2r_ladder", "analog_output_buffer"]}),
        ("no-mcu", "No microcontroller is fitted.", "part_class_exact", {"part_class": "microcontroller", "value": 0}),
    ),
    "thermocouple-amp": (
        ("max31855", "A MAX31855 realization has K-type support circuitry.", "part_identity", {"identity": "MAX31855"}),
        ("terminal", "The thermocouple input uses a screw terminal.", "part_class_count", {"part_class": "screw_terminal", "minimum": 1}),
        ("spi", "A complete SPI header and valid supply/interface levels are evidenced.", "net_paths", {"paths": ["spi_header", "thermocouple_supply"]}),
    ),
    "speaker-crossover": (
        ("two-way-passive", "A passive two-way crossover topology is present.", "net_paths", {"paths": ["low_pass", "high_pass"]}),
        ("physical-parts", "Air-core inductor, film capacitors, and binding posts are fitted.", "part_classes", {"required": {"air_core_inductor": 1, "film_capacitor": 1, "binding_post": 2}}),
        ("response", "Unit-aware crossover response is checked against recorded load and crossover assumptions.", "numeric_range", {"fact": "crossover_hz"}),
        ("no-active", "No active part is fitted.", "part_class_exact", {"part_class": "active_device", "value": 0}),
    ),
    "usb-pd-trigger": (
        ("usb-c-input", "A physical USB-C PD input is fitted.", "part_class_count", {"part_class": "usb_c_receptacle", "minimum": 1}),
        ("selector-modes", "9 V, 12 V, and 20 V switch configurations are each evidenced.", "set_members", {"fact": "pd_selector_voltages", "required": [9, 12, 20]}),
        ("power-path", "Each selector mode has a real powered controller-to-output path.", "net_paths", {"paths": ["usb_c_vbus_to_pd_controller", "pd_controller_to_output"]}),
        ("ratings", "Power-path components are rated for every selected output mode.", "numeric_range", {"fact": "power_path_voltage_rating_v", "minimum": 20}),
    ),
    "usb-c-full-breakout": (
        ("receptacle", "A real USB-C receptacle is fitted.", "part_class_count", {"part_class": "usb_c_receptacle", "minimum": 1}),
        ("contacts", "VBUS, GND, CC, SBU, and every SuperSpeed lane map distinctly to 0.1-inch headers.", "connector_map", {"connector": "usb_c_to_header", "required": ["VBUS", "GND", "CC1", "CC2", "SBU1", "SBU2", "TX1P", "TX1N", "RX1P", "RX1N", "TX2P", "TX2N", "RX2P", "RX2N"], "distinct": True}),
        ("fine-pitch-escape", "Delivered geometry proves legal fine-pitch escape routing.", "gate", {"gate": "geometry"}),
    ),
    "usb-a-power-splitter": (
        ("ports", "USB-C input and two physical USB-A power outputs are fitted.", "part_classes", {"required": {"usb_c_receptacle": 1, "usb_a_connector": 2}}),
        ("independent-limits", "Each USB-A output has independent current limiting.", "channel_count", {"channel": "independent_current_limit", "minimum": 2}),
        ("status-leds", "Meaningful status LED circuits are present.", "channel_count", {"channel": "status_led_circuit", "minimum": 2}),
        ("power-budget", "The input/output power budget is recorded and satisfied.", "numeric_range", {"fact": "power_budget_margin_w", "minimum": 0}),
    ),
    "rs485-terminal": (
        ("max485", "A MAX485 transceiver realization is fitted.", "part_identity", {"identity": "MAX485"}),
        ("isolation", "Required signal and power isolation preserve separate reference domains.", "net_paths", {"paths": ["isolated_signal_path", "isolated_power_path"]}),
        ("terminals", "A/B/GND use screw terminals.", "channel_count", {"channel": "rs485_terminal", "minimum": 3}),
        ("dere", "The DE/RE jumper operates on the intended control path.", "net_paths", {"paths": ["de_re_jumper"]}),
    ),
    "stm32-min": (
        ("stm32-package", "STM32F103 is in LQFP-48.", "part_identity", {"identity": "STM32F103", "package": "LQFP-48"}),
        ("usb-crystal-swd", "USB-C, 8 MHz crystal, and accessible SWD are connected.", "net_paths", {"paths": ["usb_c", "crystal_8mhz", "swd"]}),
        ("boot-reset", "Boot/reset buttons have reviewed support circuitry and valid pin bindings.", "net_paths", {"paths": ["boot_button", "reset_button"]}),
        ("programming", "A usable programming path is evidenced.", "gate", {"gate": "programming"}),
    ),
    "rp2040-min": (
        ("rp2040-package", "RP2040 is in QFN-56.", "part_identity", {"identity": "RP2040", "package": "QFN-56"}),
        ("core-support", "QSPI flash, USB-C, and 12 MHz crystal are connected.", "net_paths", {"paths": ["qspi_flash", "usb_c", "crystal_12mhz"]}),
        ("castellations", "GPIO uses usable castellated edge geometry, not interior pads.", "gate", {"gate": "castellations"}),
        ("programming", "A usable programming path is evidenced.", "gate", {"gate": "programming"}),
    ),
    "fpc-breakout": (
        ("fpc", "A real 24-contact 0.5 mm-pitch FPC/FFC connector is fitted.", "part_class_count", {"part_class": "fpc_0p5mm_24", "minimum": 1}),
        ("contacts", "All 24 contacts map distinctly to the 0.1-inch header row.", "connector_map", {"connector": "fpc24_to_header", "required": [str(pin) for pin in range(1, 25)], "distinct": True}),
        ("ownership", "Connector and header hardware have exactly one physical owner.", "gate", {"gate": "ownership"}),
    ),
    "esp32-s3-sensor": (
        ("parts", "ESP32-S3 and BME280 temperature/humidity/pressure interface are fitted.", "part_identities", {"identities": ["ESP32-S3", "BME280"]}),
        ("usb-programming", "Physical USB-C and usable programming access are connected.", "net_paths", {"paths": ["usb_c", "programming"]}),
        ("status-led", "A correctly connected, current-limited status LED is fitted.", "net_paths", {"paths": ["status_led"]}),
    ),
    "nrf52-beacon": (
        ("reviewed-nrf", "A reviewed nRF52840 package/order code is fitted.", "part_identity", {"identity": "nRF52840"}),
        ("rf", "RF support and a physical chip antenna are connected.", "net_paths", {"paths": ["rf_antenna"]}),
        ("antenna-geometry", "Delivered antenna placement and copper clearance follow the reviewed chip-antenna reference.", "gate", {"gate": "rf_antenna_geometry"}),
        ("user-power", "Coin-cell holder and user button are fitted.", "part_classes", {"required": {"coin_cell_holder": 1, "button": 1}}),
        ("programming", "A usable programming path is evidenced.", "gate", {"gate": "programming"}),
    ),
    "lora-node": (
        ("radio", "SX1276 module and real SMA RF path are fitted.", "part_identity", {"identity": "SX1276"}),
        ("sma", "A physical SMA connector is on the RF path.", "part_class_count", {"part_class": "sma_connector", "minimum": 1}),
        ("mcu-terminal", "Reviewed STM32L0, screw-terminal sensor interface, programming, and power support are connected.", "net_paths", {"paths": ["stm32l0", "sensor_terminal", "programming", "power"]}),
    ),
    "buck-3a": (
        ("input-voltage", "The input operating point is 5 V and is inside the selected device operating range.", "numeric_range", {"fact": "input_voltage_v", "minimum": 5, "maximum": 5}),
        ("output-voltage", "The output operating point is 3.3 V.", "numeric_range", {"fact": "output_voltage_v", "minimum": 3.3, "maximum": 3.3}),
        ("output-current", "The output path is rated for at least 3 A.", "numeric_range", {"fact": "output_current_rating_a", "minimum": 3}),
        ("terminals", "Input and output use screw terminals.", "part_class_count", {"part_class": "screw_terminal", "minimum": 2}),
        ("thermal", "Thermal-via copper is present in delivered geometry.", "gate", {"gate": "thermal_geometry"}),
        ("converter", "TPS5430 and its support circuit are evidenced.", "part_identity", {"identity": "TPS5430"}),
    ),
    "highside-switch-10a": (
        ("mosfet-path", "A logic-controlled P-channel high-side MOSFET path is present.", "net_paths", {"paths": ["highside_mosfet_path"]}),
        ("load-terminal", "The load connects through a physical screw terminal.", "part_class_count", {"part_class": "screw_terminal", "minimum": 1}),
        ("terminals-rating", "Screw terminals, conductors, and thermal features are rated for declared 10 A conditions.", "numeric_range", {"fact": "load_current_rating_a", "minimum": 10}),
        ("gate-drive", "Gate-drive limits are within the selected MOSFET limits.", "numeric_range", {"fact": "gate_drive_margin_v", "minimum": 0}),
        ("thermal", "Real thermal geometry exists without fictitious signal nets.", "gate", {"gate": "thermal_geometry"}),
    ),
    "led-cc-driver": (
        ("usb-c", "A USB-C power input is fitted.", "part_class_count", {"part_class": "usb_c_receptacle", "minimum": 1}),
        ("current-path", "The nominal constant-current setpoint is 1 A over the declared single-LED envelope.", "numeric_range", {"fact": "led_current_a", "minimum": 1, "maximum": 1}),
        ("topology", "Current-setting/feedback and switcher support topology are valid.", "net_paths", {"paths": ["current_feedback", "switcher_support"]}),
        ("thermal-no-mcu", "Heatsink copper exists and no MCU is fitted.", "part_class_exact", {"part_class": "microcontroller", "value": 0}),
        ("thermal-geometry", "Delivered copper/heatsink geometry satisfies the declared thermal feature.", "gate", {"gate": "thermal_geometry"}),
    ),
    "dual-rail-supply": (
        ("conversion", "A real 24 V to +12 V and -12 V DC-DC conversion is present.", "net_paths", {"paths": ["24v_to_plus12", "24v_to_minus12"]}),
        ("terminals", "Actual screw-terminal outputs and connected output filters are fitted.", "net_paths", {"paths": ["plus12_output_filter", "minus12_output_filter"]}),
        ("ratings", "Ratings are checked against recorded load assumptions.", "numeric_range", {"fact": "load_margin_w", "minimum": 0}),
    ),
    "relay-quad": (
        ("relays", "Four through-hole relays are fitted.", "part_class_count", {"part_class": "through_hole_relay", "minimum": 4}),
        ("drive", "An SMT ULN2003 provides reviewed coil/flyback drive paths.", "net_paths", {"paths": ["uln2003_coil_drive", "relay_flyback"]}),
        ("isolation", "Inputs are genuinely opto-isolated across their domains.", "net_paths", {"paths": ["opto_isolated_input"]}),
        ("outputs", "Screw-terminal contact outputs are fitted.", "part_class_count", {"part_class": "screw_terminal", "minimum": 4}),
    ),
    "encoder-oled-panel": (
        ("encoder", "A through-hole rotary encoder with push button is fitted.", "part_class_count", {"part_class": "through_hole_rotary_encoder", "minimum": 1}),
        ("oled-buttons", "An SMT I2C OLED and three additional buttons are independently connected.", "channel_count", {"channel": "additional_button", "minimum": 3}),
        ("oled", "A physical SMT I2C OLED is fitted and connected to the I2C bus.", "part_class_count", {"part_class": "smt_i2c_oled", "minimum": 1}),
        ("oled-i2c", "The OLED I2C bus path is physically connected.", "net_paths", {"paths": ["oled_i2c"]}),
        ("button-connections", "Each additional button has an independent delivered electrical path.", "net_paths", {"paths": ["additional_button_connections"]}),
        ("mounting", "Four physical mounting holes are present.", "geometry_count", {"feature": "mounting_hole", "minimum": 4}),
    ),
    "proto-shield": (
        ("uno-geometry", "Canonical Arduino Uno shield geometry and pin mapping are present.", "gate", {"gate": "uno_shield_geometry"}),
        ("stacking", "Real stacking through-hole headers have explicit ownership.", "part_class_count", {"part_class": "arduino_stacking_header", "minimum": 2}),
        ("proto-regulator", "Usable prototyping area and SMT 3.3 V regulator are present.", "net_paths", {"paths": ["3v3_regulator"]}),
        ("prototyping-area", "The delivered board has a usable, verified prototyping area.", "gate", {"gate": "prototyping_area"}),
        ("regulator", "An SMT 3.3 V regulator is physically fitted.", "part_class_count", {"part_class": "smt_3v3_regulator", "minimum": 1}),
    ),
    "esp32-dual-motor": (
        ("parts", "ESP32-S3 and two DRV8833 devices are fitted.", "part_class_count", {"part_class": "drv8833", "minimum": 2}),
        ("esp32-s3", "A reviewed ESP32-S3 is fitted.", "part_identity", {"identity": "ESP32-S3"}),
        ("motor-channels", "Both devices expose usable motor channels and screw terminals.", "channel_count", {"channel": "motor_channel", "minimum": 4}),
        ("buck", "Buck conversion supports the declared 2S battery envelope.", "numeric_range", {"fact": "input_voltage_max_v", "minimum": 8.4}),
        ("programming", "A usable programming path is evidenced.", "gate", {"gate": "programming"}),
    ),
    "can-node": (
        ("parts", "STM32 and SN65HVD230 are fitted.", "part_identities", {"identities": ["STM32", "SN65HVD230"]}),
        ("db9", "DB9 has the reviewed CAN pin mapping.", "connector_map", {"connector": "db9_can", "required": ["CANH", "CANL", "GND"], "distinct": True}),
        ("termination", "CAN termination is genuinely switchable.", "net_paths", {"paths": ["switchable_can_termination"]}),
        ("support", "Supply/logic levels and programming path are supported.", "gate", {"gate": "programming"}),
    ),
    "daq-8ch": (
        ("adc", "An MCU and two ADS1115 devices have compatible I2C addresses.", "part_class_count", {"part_class": "ADS1115", "minimum": 2}),
        ("mcu", "A reviewed MCU is fitted.", "part_class_count", {"part_class": "microcontroller", "minimum": 1}),
        ("adc-addresses", "The two ADS1115 devices use compatible distinct I2C addresses.", "net_paths", {"paths": ["ads1115_i2c_addresses"]}),
        ("inputs", "Eight distinct analog inputs reach screw terminals.", "connector_map", {"connector": "analog_inputs", "required": [f"AI{channel}" for channel in range(1, 9)], "distinct": True}),
        ("range", "Input range is valid and USB-C/power/programming are implemented.", "net_paths", {"paths": ["input_range", "usb_c", "programming"]}),
    ),
    "gpio-expander": (
        ("mcp23017", "An MCP23017 is fitted.", "part_identity", {"identity": "MCP23017"}),
        ("gpio", "All sixteen GPIOs reach screw terminals.", "connector_map", {"connector": "gpio_terminals", "required": [f"GPIO{pin}" for pin in range(1, 17)], "distinct": True}),
        ("i2c", "Physical chainable I2C interface and valid address/pull-up/supply choices are evidenced.", "net_paths", {"paths": ["chainable_i2c"]}),
    ),
    "servo-driver-16": (
        ("pca9685", "A PCA9685 is fitted.", "part_identity", {"identity": "PCA9685"}),
        ("headers", "Sixteen independently mapped signal/power/ground servo headers are along the edge.", "connector_map", {"connector": "servo_headers", "required": [f"SERVO{channel}" for channel in range(1, 17)], "distinct": True}),
        ("headers-edge", "All servo headers are physically placed along the delivered board edge.", "gate", {"gate": "servo_headers_edge"}),
        ("header-power-ground", "Every servo header has delivered power and ground connectivity.", "net_paths", {"paths": ["servo_header_power_ground"]}),
        ("power", "Power screw terminal and declared external-load distribution budget are evidenced.", "net_paths", {"paths": ["servo_power_terminal"]}),
    ),
    "stepper-a4988": (
        ("a4988", "A4988 support and current-setting circuit are present.", "net_paths", {"paths": ["a4988_support", "current_setting"]}),
        ("switches", "Microstep-select DIP switches operate on the intended pins.", "net_paths", {"paths": ["microstep_dip"]}),
        ("motor-power", "Motor connector and 12 V screw terminal are fitted.", "part_class_count", {"part_class": "screw_terminal", "minimum": 1}),
        ("terminal-map", "Motor and 12 V terminal mappings are complete.", "connector_map", {"connector": "stepper_terminals", "required": ["MOTOR_A1", "MOTOR_A2", "MOTOR_B1", "MOTOR_B2", "VIN12", "GND"], "distinct": True}),
        ("limits", "Pin capabilities and thermal/current envelope are valid.", "numeric_range", {"fact": "motor_current_margin_a", "minimum": 0}),
    ),
    "audio-jack-buffer": (
        ("jacks", "Exactly four physical 3.5 mm jacks are fitted.", "part_class_exact", {"part_class": "audio_jack_3p5mm", "value": 4}),
        ("jacks-edge", "All four physical audio jacks are placed along the delivered board edge.", "gate", {"gate": "audio_jacks_edge"}),
        ("channels", "Declared channels map completely into unity-gain op-amp buffering.", "channel_count", {"channel": "audio_buffer_channel", "minimum": 4}),
        ("range", "Supply and signal range are valid.", "numeric_range", {"fact": "audio_headroom_v", "minimum": 0}),
        ("no-mcu", "No microcontroller is fitted.", "part_class_exact", {"part_class": "microcontroller", "value": 0}),
    ),
    "round-led-ring": (
        ("outline", "The delivered outline is a 60 mm circle.", "outline", {"shape": "circle", "diameter_mm": 60}),
        ("leds", "Twelve WS2812B LEDs are evenly spaced in a circle with complete data chain.", "channel_count", {"channel": "ws2812b", "minimum": 12}),
        ("led-placement", "The WS2812B LEDs are evenly spaced on the delivered circular geometry.", "gate", {"gate": "ws2812b_even_circle"}),
        ("data-chain", "The delivered WS2812B data chain reaches all twelve LEDs.", "net_paths", {"paths": ["ws2812b_data_chain"]}),
        ("controller-power", "ATtiny412/programming and JST-PH power are fitted.", "net_paths", {"paths": ["attiny412_programming", "jst_ph_power"]}),
        ("no-edge", "No edge connector is present.", "part_class_exact", {"part_class": "edge_connector", "value": 0}),
    ),
    "rounded-c3-devboard": (
        ("controller", "A reviewed ESP32-C3 is fitted.", "part_identity", {"identity": "ESP32-C3"}),
        ("outline", "The delivered outline has rounded corners.", "outline", {"shape": "rounded_rect"}),
        ("edge-interfaces", "USB-C is on one edge and real 2x10 0.1-inch GPIO header on the opposite edge.", "net_paths", {"paths": ["usb_c_edge", "gpio_header_opposite_edge"]}),
        ("gpio-programming", "Mapped GPIO and programming access are usable.", "gate", {"gate": "programming"}),
    ),
    "chamfered-badge": (
        ("outline", "The delivered outline is chamfered.", "outline", {"shape": "chamfered_rect"}),
        ("parts", "ATtiny1614, six 0805 LEDs, and CR2032 holder are fitted.", "part_classes", {"required": {"ATtiny1614": 1, "led_0805": 6, "coin_cell_holder": 1}}),
        ("touch", "Two real capacitive touch pads use touch-capable pins.", "channel_count", {"channel": "capacitive_touch_pad", "minimum": 2}),
        ("budget", "Current budget is valid.", "numeric_range", {"fact": "battery_current_margin_a", "minimum": 0}),
    ),
    "hex-env-sensor": (
        ("outline", "The delivered outline is hexagonal.", "outline", {"shape": "hexagon"}),
        ("sensor", "BME280 temperature/humidity/pressure interface is fitted.", "part_identity", {"identity": "BME280"}),
        ("qwiic", "A real compatible Qwiic connector/interface is fitted.", "part_class_count", {"part_class": "qwiic_connector", "minimum": 1}),
        ("led", "Power LED operates from a complete circuit.", "net_paths", {"paths": ["power_led"]}),
    ),
    "star-ornament": (
        ("outline", "The delivered outline is a star with a top hang hole.", "outline", {"shape": "star"}),
        ("leds", "Five warm-white LEDs occupy the five points.", "channel_count", {"channel": "warm_white_led", "minimum": 5}),
        ("led-point-placement", "Each warm-white LED is located at a distinct delivered star point.", "gate", {"gate": "star_point_led_placement"}),
        ("controller-power", "ATtiny402 and CR2032 power are fitted.", "part_classes", {"required": {"ATtiny402": 1, "coin_cell_holder": 1}}),
        ("hang-hole", "A physical top hang hole is present.", "geometry_count", {"feature": "hang_hole", "minimum": 1}),
        ("budget-programming", "LED/battery budget and programming access are valid.", "gate", {"gate": "programming"}),
    ),
    "snowman-ornament": (
        ("outline", "The delivered outline has three stacked snowman sections.", "outline", {"shape": "snowman"}),
        ("leds", "Warm-white LEDs occupy base, body, and head sections.", "channel_count", {"channel": "warm_white_led", "minimum": 3}),
        ("controller-power", "ATtiny402 and CR2032 power are fitted.", "part_classes", {"required": {"ATtiny402": 1, "coin_cell_holder": 1}}),
        ("sections", "The illuminated sections are base, body, and head.", "set_members", {"fact": "snowman_led_sections", "required": ["base", "body", "head"]}),
        ("budget-programming", "Orderable hardware, current budget, and programming access are valid.", "gate", {"gate": "programming"}),
    ),
}


# A compiler-boundary reference is reviewed before synthesis and routing exist.
# It can honestly prove only checks whose facts come from reviewed BOM, wiring,
# recipe, and engineering-calculation inputs.  Every other check needs a
# delivered board or build report and MUST be deferred to campaign fulfillment
# explicitly; a reference never records an artifact gate as passing.
REFERENCE_CHECKABLE_KINDS = frozenset({
    "part_class_count", "part_class_exact", "part_classes",
    "part_identity", "part_identities", "part_inventory",
    "net_paths", "numeric_range", "set_members", "channel_count",
})


def reference_obligation_ids(slug: str, version: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split one contract's mandatory obligations into reference and deferred ids."""
    reference: list[str] = []
    deferred: list[str] = []
    for obligation in CONTRACTS_BY_VERSION[version][slug]["obligations"]:
        target = reference if obligation["check"]["kind"] in REFERENCE_CHECKABLE_KINDS else deferred
        target.append(obligation["id"])
    return tuple(reference), tuple(deferred)


_CHECK_OWNERS = {
    "part_class_count": ("bom/part realization", "committed BOM and KiCad board footprints"),
    "part_class_exact": ("bom/part realization", "committed BOM and KiCad board footprints"),
    "part_classes": ("bom/part realization", "committed BOM and KiCad board footprints"),
    "part_identity": ("part identity/sourceability", "committed BOM and manufacturer source"),
    "part_identities": ("part identity/sourceability", "committed BOM and manufacturer source"),
    "part_inventory": ("part identity/sourceability", "committed BOM, source records, and ratings"),
    "pin_mapping": ("symbol/footprint realization", "symbol, footprint, and board-pad mapping"),
    "net_paths": ("wiring/synthesis", "schematic/board connectivity artifacts"),
    "numeric_range": ("electrical review", "recorded calculation and component operating limits"),
    "connector_map": ("wiring/synthesis", "connector pin/net mapping from schematic and delivered board"),
    "set_members": ("wiring/synthesis", "schematic/board connectivity artifacts"),
    "channel_count": ("wiring/synthesis", "schematic/board connectivity artifacts"),
    "geometry_count": ("placement/routing", "delivered KiCad board geometry"),
    "outline": ("placement/routing", "delivered KiCad board outline"),
    "gate": ("build acceptance", "ERC/DRC/programming/geometry report"),
    "applicable_gate": ("build acceptance", "applicability record and gate report"),
    "artifacts": ("fabrication export", "delivered fabrication artifacts"),
}


def _obligation(slug: str, suffix: str, statement: str, check: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    owner, evidence_source = _CHECK_OWNERS[check]
    return {
        "id": f"{slug}.{suffix}", "statement": statement, "check": {"kind": check, **parameters},
        "owner": owner, "evidence_source": evidence_source,
    }


def _contract(entry: Mapping[str, str], index: int, version: str) -> dict[str, Any]:
    slug = entry["slug"]
    specifics = _SPECIFIC[slug]
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "version": version,
        "index": index,
        "slug": slug,
        "original_brief": entry["brief"],
        "execution_brief": entry["brief"],
        "feasibility": {"status": "not_yet_reviewed", "reason": "No reviewed realizability evidence is recorded."},
        "sourceability": {"status": "unverified", "reason": "No reviewed MPN, supplier, package, and rating evidence is recorded."},
        "consent": [],
        "operating_limits": [{"status": "unverified", "reason": "Operating limits and any open engineering assumptions require review."}],
        "obligations": [
            *[_obligation(slug, suffix, statement, check, parameters) for suffix, statement, check, parameters in specifics],
            *[_obligation(slug, suffix, statement, check, parameters) for suffix, statement, check, parameters in COMMON_OBLIGATIONS],
        ],
    }


def _build_contracts(version: str) -> dict[str, dict[str, Any]]:
    return {entry["slug"]: _contract(entry, index, version) for index, entry in enumerate(BENCHMARK_PROMPTS, 1)}


ORIGINAL_CONTRACTS = _build_contracts(ORIGINAL_CORPUS_VERSION)


# Reviewed per-brief dispositions.  Every status below is derived from the
# reference rows in tests/fixtures/reference_inputs/*.json that
# `python -m kicraft.eval.design_acceptance --references` validates: a slug
# whose row is absent, or whose row does not validate, keeps the
# not_yet_reviewed/unverified default instead of a claim.  A reference row can
# only prove the obligations its reviewed architecture, recipe/lowerer, BOM and
# wiring inputs settle; delivered-board gates are recorded as deferred and are
# never claimed from a reference.  Every reason names the row, the obligation
# ids, the reviewed part identities or order codes, and the manufacturer or
# catalog source it relies on.  Consent records stay empty here: only the
# user-approved 5 V buck variant below carries one, and the original buck-3a
# specification conflict plus the speaker-crossover sourcing block keep their
# own frozen records immediately below this table.
_REVIEWED_DISPOSITIONS: dict[str, dict[str, Any]] = {
    "usb-pd-trigger": {
        "operating_limits": [
            {
                "part": "TYPE-C-31-M-12",
                "source": "https://lcsc.com/product-detail/USB-Type-C_Korean-Hroparts-Elec-TYPE-C-31-M-12_C165948.html",
                "limit": "20 V, 5 A, 16 contacts, 10 000 mating cycles, -30 to +80 C",
                "disposition": "reviewed receptacle; the 20 V rating is what the usb-pd-trigger.ratings obligation relies on for the 20 V selector mode"
            },
            {
                "part": "CH224K",
                "source": "https://www.wch-ic.com/downloads/CH224DS1_PDF.html",
                "limit": "4-30 V input, 100 W PD3.0 sink; VDD 3.0-3.6 V; high-voltage pin limit 13.5 V through the reviewed series resistors; -40 to +90 C",
                "disposition": "reviewed PD sink controller covering the 9 V, 12 V and 20 V negotiated modes (usb-pd-trigger.selector-modes)"
            },
            {
                "part": "SS13D07VG4",
                "source": "https://www.lcsc.com/datasheet/C2681578.pdf",
                "limit": "50 V DC, 0.5 A, SP3T slide switch, 5 000 cycles, -10 to +50 C",
                "disposition": "reviewed voltage selector switching CFG1 between the reviewed 6.8 kohm, 24 kohm and open straps"
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "The reference row validates its reviewed obligations, but its recorded boundary run (logs/self_eval/remediation_20260916T013532Z/reference_rows/group-a/evidence/usb-pd-trigger-boundary/boundary_result.json, limit wiring) was rejected at commit_bom by the 9.26 dual-inventory gate: U1 CH224K (LCSC C970725) is out of stock at the lcsc.com retail storefront (0 available, min buy 1), so no committed wiring/board artifact exists for this brief and the deterministic BOM/wiring stage commit is not evidenced."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reviewed row records that the 9.26 dual-inventory gate rejects U1 CH224K (LCSC C970725, catalog stock 7051) because the lcsc.com retail storefront shows 0 available, so current orderable stock is not evidenced."
        }
    },
    "rc-lowpass-bnc": {
        "operating_limits": [
            {
                "part": "KH-BNC50-3511",
                "source": "https://www.lcsc.com/datasheet/C2837587.pdf",
                "limit": "50 ohm, DC-3 GHz, through-hole board-side jack",
                "disposition": "reviewed connector for both BNC ports (rc-lowpass-bnc.bnc-count)"
            },
            {
                "part": "3296W-1-103LF",
                "source": "https://www.bourns.com/docs/product-datasheets/3296.pdf",
                "limit": "10 kohm +/-10 %, 0.5 W at 70 C, 25-turn, 300 V max",
                "disposition": "reviewed rheostat; 10 kohm with the reviewed 10 nF C0G sets f_min = 1/(2*pi*R*C) = 1591.5 Hz (rc-lowpass-bnc.adjustable-response)"
            },
            {
                "part": "C0805C103J5GACTU",
                "source": "https://search.kemet.com/download/specsheet/C0805C103J5GACTU",
                "limit": "10 nF +/-5 %, 50 V, C0G/NP0",
                "disposition": "reviewed filter capacitor held fixed while the trimmer adjusts the corner"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/benchmark_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations bnc-count, trim-pot, adjustable-response, no-mcu, sourceable-parts pass over reviewed part identities (KH-BNC50-3511, 3296W-1-103LF, C0805C103J5GACTU) resolved by adjustable-rc-lowpass@1, bnc-connector@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Kinghelm KH-BNC50-3511 (LCSC C2837587, catalog stock 4742 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Bourns 3296W-1-103LF (LCSC C34846, catalog stock 10605 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); KEMET C0805C103J5GACTU (LCSC C2167597, catalog stock 5267 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "r2r-dac": {
        "operating_limits": [
            {
                "part": "MCP6001T-I/OT",
                "source": "https://lcsc.com/product-detail/Low-Power-OpAmps_MICROCHIP_MCP6001T-I-OT_MCP6001T-I-OT_C116490.html",
                "limit": "1.8-6 V supply, 1 MHz GBW, rail-to-rail I/O, 100 uA quiescent",
                "disposition": "reviewed output buffer; 5 V operation covers the recorded full-scale 3.3 V ladder output (r2r-dac.buffered-output)"
            },
            {
                "part": "FRC0603F1002TS / 0603WAF2002T5E",
                "source": "https://jlcpcb.com/partdetail/FOJAN-FRC0603F1002TS/C2906982",
                "limit": "10 kohm and 20 kohm +/-1 %, 100 mW, 75 V",
                "disposition": "reviewed R-2R ladder elements; the reviewed ratio defines the eight-bit ramp (r2r-dac.r2r-network)"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/benchmark_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations logic-inputs, r2r-network, buffered-output, no-mcu, sourceable-parts pass over reviewed part identities (MCP6001T-I/OT, FRC0603F1002TS, 0603WAF2002T5E, GRM188R71C104KA01D, 2.54-1*40P) resolved by mcp6001-follower@1, pin-header@1, r2r-ladder@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Microchip MCP6001T-I/OT (LCSC C116490, catalog stock 140066 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); FOJAN FRC0603F1002TS (LCSC C2906982, catalog stock 10334802 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); UNI-ROYAL 0603WAF2002T5E (LCSC C4184, catalog stock 2258874 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); BOOMELE 2.54-1*40P header strip (LCSC C2337, catalog stock 87161 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "speaker-crossover": {
        "operating_limits": [
            {
                "part": "LW18-50",
                "source": "https://www.daytonaudio.com/product/1327/lw18-50-0-50mh-18-awg-perfect-layer-inductor",
                "limit": "0.50 mH +/-3 %, 18 AWG air-core, 0.33 ohm DCR, 300 W RMS",
                "disposition": "reviewed low-pass inductor; the original procurement gate conflict remains unresolved, so sourceability stays sourcing_blocked"
            },
            {
                "part": "MKP20685J2G362230",
                "source": "kicraft/parts_library/kyet-mkp20685j2g362230/manifest.json",
                "limit": "6.8 uF +/-5 %, 400 V DC metallized polypropylene film",
                "disposition": "reviewed high-pass capacitor in the reviewed 7.8 uF high-pass bank"
            },
            {
                "part": "MKP1848510924K2",
                "source": "kicraft/parts_library/vishay-mkp1848510924k2/manifest.json",
                "limit": "1.0 uF +/-5 %, 1200 V DC metallized polypropylene film",
                "disposition": "reviewed high-pass capacitor paralleled with the reviewed 6.8 uF part"
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "The reference row does not validate against its contract: speaker-crossover.response records no declared range for crossover_hz, speaker-crossover.sourceable-parts is recorded as failed, and the row's deferred set and obligation ids do not match the contract's, so no reviewed boundary commit is evidenced."
        }
    },
    "buck-3a": {
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: TI TPS5430DDAR (LCSC C9864, catalog stock 165665 matched in the catalog by exact-MPN lookup for the reviewed row order code (SOIC-8-EP 5.5-36 V / 3 A TPS5430), verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "highside-switch-10a": {
        "operating_limits": [
            {
                "part": "AONR21357",
                "source": "https://www.lcsc.com/product-detail/MOSFETs_Alpha-Omega-Semicon-AONR21357_C431196.html",
                "limit": "VDS 30 V; continuous drain 34 A; RDS(on) 12.3 mOhm at VGS -4.5 V; VGS +/-25 V; 30 W package dissipation",
                "disposition": "reviewed load switch; the declared 10 A load and reviewed gate drive stay inside the ratings"
            },
            {
                "part": "BZT52C10",
                "source": "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8756679576573890560",
                "limit": "10 V zener, 9.5-10.5 V breakdown, 500 mW",
                "disposition": "reviewed gate clamp keeping VGS inside the reviewed +/-25 V MOSFET limit"
            },
            {
                "part": "WJ126V-5.0-02P-14-00A",
                "source": "https://lcsc.com/product-detail/Terminal-Blocks_WJ126V-5-0-2P_C8404.html",
                "limit": "250 V, 18 A, 14-26 AWG, -40 to +105 C",
                "disposition": "reviewed load terminal rated above the declared 10 A condition"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/benchmark_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations mosfet-path, load-terminal, terminals-rating, gate-drive, sourceable-parts pass over reviewed part identities (AONR21357, WJ126V-5.0-02P-14-00A, RC0603FR-07100KL, AC0603FR-071KL, BZT52C10, MMBT3904, FRC0603F1002TS) resolved by screw-terminal@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (thermal, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: AOS AONR21357 (LCSC C431196, catalog stock 31744 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); KANGNEX WJ126V-5.0-02P-14-00A (LCSC C8404, catalog stock 68526 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "led-cc-driver": {
        "operating_limits": [
            {
                "part": "AL8860MP-13",
                "source": "https://www.diodes.com/datasheet/download/AL8860.pdf",
                "limit": "VIN 4.5-40 V; continuous output current 1.5 A (MSOP-8EP); 1 MHz maximum switching frequency; OTP at 150 C",
                "disposition": "reviewed driver; the USB-C 5 V input and the 1 A setpoint are inside the reviewed range (led-cc-driver.current-path)"
            },
            {
                "part": "AL8860MP-13 SET sense threshold",
                "source": "kicraft/design/part_identity.py",
                "limit": "sense_voltage_v 0.1 with 0.04 tolerance at the SET pin against VIN",
                "disposition": "reviewed sense voltage; the reviewed 0.100 ohm RSET gives the brief nominal 1 A setpoint"
            },
            {
                "part": "WSL2512R1000FEA",
                "source": "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588893225277276160",
                "limit": "0.100 ohm +/-1 %, 1 W, +/-75 ppm/C",
                "disposition": "reviewed sense resistor; 1 A dissipates 0.1 W"
            },
            {
                "part": "SMDRI127-220MT",
                "source": "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588918807184756736",
                "limit": "22 uH +/-20 %, 3.6 A rated, 7 A saturation, 43 mOhm DCR",
                "disposition": "reviewed buck inductor above the 1 A setpoint plus ripple"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/benchmark_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations usb-c, current-path, topology, thermal-no-mcu, sourceable-parts pass over reviewed part identities (AL8860MP-13, TYPE-C-31-M-12, WSL2512R1000FEA, GRM32DR71E106KA12L, SS34, SMDRI127-220MT, FRC0603F5101TS, WJ126V-5.0-02P-14-00A) resolved by usb-c-5v-sink@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (thermal-geometry, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Diodes AL8860MP-13 (LCSC C500782, catalog stock 4000 recorded in the reviewed vendored asset kicraft/parts_library/al8860/manifest.json, verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Korean Hroparts TYPE-C-31-M-12 (LCSC C165948, catalog stock 89797 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); KANGNEX WJ126V-5.0-02P-14-00A (LCSC C8404, catalog stock 68526 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "dual-rail-supply": {
        "operating_limits": [
            {
                "part": "WRA2412S-3WR2",
                "source": "https://www.lcsc.com/datasheet/C20617261.pdf",
                "limit": "18-36 VDC input (24 V nominal); +12 V and -12 V at 12.5-125 mA per rail; 3 W total; 1.5 kVDC isolation",
                "disposition": "reviewed converter covers the brief 24 V input and the recorded load assumption (dual-rail-supply.conversion)"
            },
            {
                "part": "37205000001",
                "source": "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588891356407713792",
                "limit": "500 mA, 250 V slow-blow, 35 A breaking capacity",
                "disposition": "reviewed input fuse ahead of the 24 V rail"
            },
            {
                "part": "JBLH2101M050C120RLM 100UF 50V",
                "source": "https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8602919826839531520",
                "limit": "100 uF +/-20 %, 50 V radial electrolytic",
                "disposition": "reviewed bulk capacitor on the 24 V input, inside the WRA2412 470 uF per-output limit budget"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/benchmark_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations conversion, terminals, ratings, sourceable-parts pass over reviewed part identities (WRA2412S-3WR2, 37205000001, CD54-100M 10UH, VEJ100M2ATR-0607, CC0805KKX7R0BB105, RVT1H220M0605 22UF 50V, JBLH2101M050C120RLM 100UF 50V, WJ126V-5.0-02P-14-00A, WJ126V-5.0-03P-14-00A) resolved by the reviewed recipe/lowerer work units; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: ReSine WRA2412S-3WR2 (LCSC C20617261, catalog stock 254 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); KANGNEX WJ126V-5.0-03P-14-00A (LCSC C8401, catalog stock 8643 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); KANGNEX WJ126V-5.0-02P-14-00A (LCSC C8404, catalog stock 68526 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "audio-jack-buffer": {
        "operating_limits": [
            {
                "part": "MCP6004-I/SL",
                "source": "https://www.lcsc.com/datasheet/C1346056.pdf",
                "limit": "1.8-6.0 V supply, four amplifiers, rail-to-rail input/output",
                "disposition": "reviewed unity-gain buffer supplying the four reviewed audio channels (audio-jack-buffer.channels)"
            },
            {
                "part": "SJ1-3533NG",
                "source": "https://www.sameskydevices.com/product/resource/sj1-353xng.pdf",
                "limit": "12 VDC, 1 A, -25 to +85 C, no internal switches",
                "disposition": "reviewed through-hole jack at the exact four-jack quantity (audio-jack-buffer.jacks)"
            },
            {
                "part": "GRM188R61A106KE69D",
                "source": "https://www.murata.com/en-global/products/productdetail?partno=GRM188R61A106KE69D",
                "limit": "10 uF +/-10 %, 10 V, X5R, 0603",
                "disposition": "reviewed coupling capacitor; the eight 100 kohm bias resistors still have no reviewed MPN (row unresolved coverage)"
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "The reference row validates its reviewed obligations, but its own unresolved coverage records that normalization, unit ownership and commit are still required, so the deterministic BOM/wiring stage commit is not evidenced for this brief."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Same Sky SJ1-3533NG (LCSC C4992459, catalog stock 323 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Microchip MCP6004-I/SL (LCSC C1346056, catalog stock 588 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "hex-env-sensor": {
        "operating_limits": [
            {
                "part": "BME280",
                "source": "https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme280-ds002.pdf",
                "limit": "1.71-3.6 V supply, -40 to +85 C, 0-100 %RH, I2C/SPI",
                "disposition": "reviewed sensor; the reviewed 3.3 V Qwiic host supply is inside the operating range"
            },
            {
                "part": "SM04B-SRSS-TB(LF)(SN)",
                "source": "https://www.jst-mfg.com/product/pdf/eng/eSH.pdf",
                "limit": "50 V / 1 A per contact, 1.00 mm pitch, 4 positions",
                "disposition": "reviewed Qwiic receptacle carrying the reviewed 3.3 V/SDA/SCL interface"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/benchmark_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations sensor, qwiic, led, sourceable-parts pass over reviewed part identities (BME280, SM04B-SRSS-TB(LF)(SN), GRM188R71C104KA01D) resolved by led-current-resistor@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (outline, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Bosch BME280 (LCSC C92489, catalog stock 10327 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); JST SM04B-SRSS-TB(LF)(SN) (LCSC C160404, catalog stock 7284 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "thermocouple-amp": {
        "operating_limits": [
            {
                "status": "unverified",
                "reason": "No reviewed operating limit is relied on for this brief yet: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed part limit this disposition can rely on."
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "Reference row tests/fixtures/reference_inputs/interface_references.json does not validate against its contract, so no reviewed boundary commit is evidenced: the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; thermocouple-amp.max31855: has no reviewed pass; thermocouple-amp.terminal: has no reviewed pass; thermocouple-amp.spi: has no reviewed pass; thermocouple-amp.sourceable-parts: has no reviewed pass."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed exact order code with current stock (the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; thermocouple-amp.max31855: has no reviewed pass; thermocouple-amp.terminal: has no reviewed pass; thermocouple-amp.spi: has no reviewed pass; thermocouple-amp.sourceable-parts: has no reviewed pass)."
        }
    },
    "usb-c-full-breakout": {
        "operating_limits": [
            {
                "part": "12401610E4-2A",
                "source": "https://cdn.amphenol-cs.com/media/wysiwyg/files/drawing/c12401610_c.pdf",
                "limit": "24 contacts, USB 3.2 receptacle (no voltage/current rating is published for this order code)",
                "disposition": "reviewed receptacle; the passive breakout maps the 16 requested contacts and claims no power role"
            },
            {
                "part": "2.54-1*40P direct-pin strip",
                "source": "https://lcsc.com/product-detail/Male-Header_2-54mm-1-40P-Straight-Headers-Pins_C2337.html",
                "limit": "3 A per contact, 2.54 mm pitch, -25 to +85 C",
                "disposition": "reviewed cut-down header segments for the mapped breakout contacts"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/interface_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations receptacle, sourceable-parts pass over reviewed part identities (12401610E4-2A, 2.54-1*40P\u76f4\u9488) resolved by the reviewed recipe/lowerer work units; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (contacts, fine-pitch-escape, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Amphenol 12401610E4-2A (LCSC C5119948, catalog stock 7394 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); BOOMELE 2.54-1*40P header strip (cut to the two 1x07 segments) (LCSC C2337, catalog stock 87161 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "usb-a-power-splitter": {
        "operating_limits": [
            {
                "part": "TPS2553DBVR",
                "source": "https://www.ti.com/lit/ds/symlink/tps2553.pdf",
                "limit": "2.5-6.5 V input, 1.5 A maximum continuous current, 85 mOhm on-resistance, -40 to +150 C",
                "disposition": "reviewed current-limited distribution switch; the two reviewed 1.25 A channels are inside the limit (usb-a-power-splitter.independent-limits)"
            },
            {
                "part": "U-A-24SS-W-2",
                "source": "https://lcsc.com/product-detail/USB-Connectors_Korean-Hroparts-Elec-U-A-24SS-W-2_C530629.html",
                "limit": "1.5 A per port, 4 contacts, USB 2.0",
                "disposition": "reviewed USB-A outputs; 1.25 A per port is inside the rating"
            },
            {
                "part": "TYPE-C-31-M-12",
                "source": "https://lcsc.com/product-detail/USB-Type-C_Korean-Hroparts-Elec-TYPE-C-31-M-12_C165948.html",
                "limit": "20 V / 5 A, 16 contacts, 10 000 mating cycles",
                "disposition": "reviewed USB-C input; the 2.5 A concurrent-load assumption still requires a qualified 3 A source (row unresolved coverage)"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/interface_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations ports, independent-limits, status-leds, power-budget, sourceable-parts pass over reviewed part identities (TYPE-C-31-M-12, U-A-24SS-W-2, TPS2553DBVR, RC0603FR-0723K2L, RC0603FR-075K1L, RC0603FR-071K5L, LTST-C190KGKT, TAJB686K010RNJ, GRM188R61A106KE69D, GRM188R71C104KA01D) resolved by usb-c-5v-sink@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Korean Hroparts TYPE-C-31-M-12 (LCSC C165948, catalog stock 89797 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Korean Hroparts U-A-24SS-W-2 (LCSC C530629, catalog stock 9522 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); TI TPS2553DBVR (LCSC C55266, catalog stock 44248 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "rs485-terminal": {
        "operating_limits": [
            {
                "part": "MAX485ESA+T",
                "source": "https://www.analog.com/media/en/technical-documentation/data-sheets/MAX1487-MAX491.pdf",
                "limit": "4.75-5.25 V supply, 2.5 Mbps, 1 driver / 1 receiver, 32 nodes, -40 to +85 C",
                "disposition": "reviewed transceiver; the reviewed 5 V logic domain is inside the operating range (rs485-terminal.max485)"
            },
            {
                "part": "ADUM1301ARWZ-RL",
                "source": "https://www.analog.com/media/en/technical-documentation/data-sheets/ADuM1300_1301.pdf",
                "limit": "2 forward / 1 reverse channel, 2500 Vrms isolation, 2.7-5.5 V per side, 1 Mbps",
                "disposition": "reviewed isolator holding GND_LOGIC and GND_FIELD separate (rs485-terminal.isolation)"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/interface_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations max485, isolation, terminals, dere, sourceable-parts pass over reviewed part identities (MAX485ESA+T, ADUM1301ARWZ-RL, B0509S-1WR3, AMS1117-5.0, WJ126V-5.0-03P-14-00A, PZ254V-11-05P, PZ254V-11-02P, RC1206FR-07680RL) resolved by the reviewed recipe/lowerer work units; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Analog Devices MAX485ESA+T (LCSC C19738, catalog stock 35473 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Analog Devices ADUM1301ARWZ-RL (LCSC C22261, catalog stock 981 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); EVISUN B0509S-1WR3 (LCSC C7500906, catalog stock 294 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "fpc-breakout": {
        "operating_limits": [
            {
                "part": "KH-FG0.5-H2.0-24PIN",
                "source": "https://www.kinghelm.net/fpc-connector-53730/55197.html",
                "limit": "24 contacts at 0.50 mm pitch, 0.3 mm FFC, bottom contact, 2 mm height, -25 to +85 C",
                "disposition": "reviewed connector matches the brief 24-contact 0.5 mm requirement; the recorded wiring boundary run still fails on header pad coverage"
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "The reference row validates its reviewed obligations, but its recorded boundary run (logs/self_eval/remediation_20260916T013532Z/reference_wiring_r1/fpc-breakout/boundary_result.json, limit wiring) failed the wiring work unit: wiring-u001 rejected the 1x24 header assignment and is missing pads J2.25-J2.36, so the deterministic BOM/wiring stage commit is not evidenced for the 24-contact contract."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Kinghelm KH-FG0.5-H2.0-24PIN (LCSC C2797213, catalog stock 2696 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); the contiguous 1x24 header is a cut-down stock 2.54 mm strip whose order code the row does not assert, so no header order code is claimed here."
        }
    },
    "relay-quad": {
        "operating_limits": [
            {
                "status": "unverified",
                "reason": "No reviewed operating limit is relied on for this brief yet: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed part limit this disposition can rely on."
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "Reference row tests/fixtures/reference_inputs/interface_references.json does not validate against its contract, so no reviewed boundary commit is evidenced: the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; relay-quad.relays: has no reviewed pass; relay-quad.drive: has no reviewed pass; relay-quad.isolation: has no reviewed pass; relay-quad.outputs: has no reviewed pass; relay-quad.sourceable-parts: has no reviewed pass."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed exact order code with current stock (the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; relay-quad.relays: has no reviewed pass; relay-quad.drive: has no reviewed pass; relay-quad.isolation: has no reviewed pass; relay-quad.outputs: has no reviewed pass; relay-quad.sourceable-parts: has no reviewed pass)."
        }
    },
    "encoder-oled-panel": {
        "operating_limits": [
            {
                "status": "unverified",
                "reason": "No reviewed operating limit is relied on for this brief yet: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed part limit this disposition can rely on."
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "Reference row tests/fixtures/reference_inputs/interface_references.json does not validate against its contract, so no reviewed boundary commit is evidenced: the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; encoder-oled-panel.encoder: has no reviewed pass; encoder-oled-panel.oled-buttons: has no reviewed pass; encoder-oled-panel.oled: has no valid result status; encoder-oled-panel.oled-i2c: has no valid result status; encoder-oled-panel.button-connections: has no valid result status; encoder-oled-panel.sourceable-parts: has no reviewed pass."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed exact order code with current stock (the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; encoder-oled-panel.encoder: has no reviewed pass; encoder-oled-panel.oled-buttons: has no reviewed pass; encoder-oled-panel.oled: has no valid result status; encoder-oled-panel.oled-i2c: has no valid result status; encoder-oled-panel.button-connections: has no valid result status; encoder-oled-panel.sourceable-parts: has no reviewed pass)."
        }
    },
    "proto-shield": {
        "operating_limits": [
            {
                "status": "unverified",
                "reason": "No reviewed operating limit is relied on for this brief yet: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed part limit this disposition can rely on."
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "Reference row tests/fixtures/reference_inputs/interface_references.json does not validate against its contract, so no reviewed boundary commit is evidenced: the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; proto-shield.stacking: has no reviewed pass; proto-shield.proto-regulator: has no reviewed pass; proto-shield.regulator: has no valid result status; proto-shield.sourceable-parts: has no reviewed pass."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed exact order code with current stock (the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; proto-shield.stacking: has no reviewed pass; proto-shield.proto-regulator: has no reviewed pass; proto-shield.regulator: has no valid result status; proto-shield.sourceable-parts: has no reviewed pass)."
        }
    },
    "can-node": {
        "operating_limits": [
            [
                "STMicroelectronics STM32F103C8T6",
                "C8734",
                270767
            ],
            "TI SN65HVD230 carries TI datasheet limits in the row but no order code, so no code is claimed for it here",
            [
                "CONNFLY DS1034-09FUNSi44 DB9 socket",
                "C77831",
                7535
            ]
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/interface_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations parts, termination, sourceable-parts pass over reviewed part identities (STM32F103C8T6, SN65HVD230, DS1034-09FUNSi44, TPS54331DDAR, TYPE-C-31-M-12) resolved by sn65hvd230-can-node@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (db9, support, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: STMicroelectronics STM32F103C8T6 (LCSC C8734, catalog stock 270767 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); TI SN65HVD230 carries TI datasheet limits in the row but no order code, so no code is claimed for it here; CONNFLY DS1034-09FUNSi44 DB9 socket (LCSC C77831, catalog stock 7535 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "daq-8ch": {
        "operating_limits": [
            {
                "status": "unverified",
                "reason": "No reviewed operating limit is relied on for this brief yet: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed part limit this disposition can rely on."
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "Reference row tests/fixtures/reference_inputs/interface_references.json does not validate against its contract, so no reviewed boundary commit is evidenced: the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; daq-8ch.adc: has no reviewed pass; daq-8ch.mcu: has no valid result status; daq-8ch.adc-addresses: has no valid result status; daq-8ch.range: has no reviewed pass; daq-8ch.sourceable-parts: has no reviewed pass."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed exact order code with current stock (the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; daq-8ch.adc: has no reviewed pass; daq-8ch.mcu: has no valid result status; daq-8ch.adc-addresses: has no valid result status; daq-8ch.range: has no reviewed pass; daq-8ch.sourceable-parts: has no reviewed pass)."
        }
    },
    "gpio-expander": {
        "operating_limits": [
            {
                "part": "MCP23017-E/SO",
                "source": "https://ww1.microchip.com/downloads/en/DeviceDoc/21952a.pdf",
                "limit": "1.8-5.5 V supply, 16 GPIO lines, I2C up to 1.7 MHz, -40 to +125 C",
                "disposition": "reviewed expander; the reviewed 3.3 V operation with A2:A0 strapped low selects 0x20 (gpio-expander.mcp23017)"
            },
            {
                "part": "ERJ-3EKF1002V",
                "source": "https://industrial.panasonic.com/cdbs/www-data/pdf/RDM0000/AOA0000C304.pdf",
                "limit": "10 kohm +/-1 %, 0603 thick-film",
                "disposition": "reviewed I2C pull-up: Rp(max) = tr(max)/(0.8473*Cb) = 11.8 kohm for a 100 pF bus, so 10 kohm is inside the bound"
            },
            {
                "part": "WJ126V-5.0-02P-14-00A",
                "source": "https://www.lcsc.com/datasheet/C8404.pdf",
                "limit": "250 V, 18 A, 5.00 mm pitch, -40 to +105 C",
                "disposition": "reviewed screw terminals for the sixteen routed GPIOs"
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "The reference row validates its reviewed obligations, but its own unresolved coverage records that the sixteen screw terminals, both I2C headers, the pull-up pair and the bypass capacitor are model-owned groups the deterministic BOM/wiring work units and the normalizer must still own, so the boundary stage commit is not evidenced."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Microchip MCP23017-E/SO (SOIC-28W) (LCSC C47023, catalog stock 1014 matched in the catalog by the SOIC-28-300mil 16-bit I2C expander signature, verified by read-only jlcparts.lookup on the local dump, 2026-09-16); KANGNEX WJ126V-5.0-02P-14-00A (LCSC C8404, catalog stock 68526 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Murata GRM188R71C104KA01D (LCSC C45000, catalog stock 3848 verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "servo-driver-16": {
        "operating_limits": [
            {
                "part": "PCA9685PW,118",
                "source": "https://www.lcsc.com/datasheet/C2678753.pdf",
                "limit": "2.3-5.5 V supply, 16 PWM channels at 12 bits, 25 mA source/sink, 6 I2C address straps",
                "disposition": "reviewed PWM driver; the reviewed 3.3 V logic and 5.5 V-tolerant outputs are inside the range (servo-driver-16.pca9685)"
            },
            {
                "part": "WJ126V-5.0-02P-14-00A",
                "source": "https://www.lcsc.com/datasheet/C8404.pdf",
                "limit": "250 V, 18 A, 5.00 mm pitch, -40 to +105 C",
                "disposition": "reviewed servo power terminal; 16 x 0.5 A = 8 A of external distribution is terminal/copper gated and never through the PCA9685"
            },
            {
                "part": "2.54-1*40P direct-pin strip",
                "source": "https://lcsc.com/product-detail/Male-Header_2-54mm-1-40P-Straight-Headers-Pins_C2337.html",
                "limit": "3 A per contact, 2.54 mm pitch",
                "disposition": "reviewed cut-down 1x03 servo headers; no reviewed manufacturer MPN is asserted for the stock land pattern"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/interface_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations pca9685, header-power-ground, power, sourceable-parts pass over reviewed part identities (PCA9685PW,118, WJ126V-5.0-02P-14-00A, 2.54-1*40P\u76f4\u9488, GRM188R71C104KA01D) resolved by connector-bank@1, pca9685-servo-bank@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (headers, headers-edge, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: NXP PCA9685PW,118 (LCSC C2678753, catalog stock 2772 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); KANGNEX WJ126V-5.0-02P-14-00A (LCSC C8404, catalog stock 68526 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); BOOMELE 2.54-1*40P header strip (cut to 1x03 servo headers) (LCSC C2337, catalog stock 87161 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "stepper-a4988": {
        "operating_limits": [
            {
                "status": "unverified",
                "reason": "No reviewed operating limit is relied on for this brief yet: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed part limit this disposition can rely on."
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "Reference row tests/fixtures/reference_inputs/interface_references.json does not validate against its contract, so no reviewed boundary commit is evidenced: the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; stepper-a4988.a4988: has no reviewed pass; stepper-a4988.switches: has no reviewed pass; stepper-a4988.motor-power: has no reviewed pass; stepper-a4988.limits: has no reviewed pass; stepper-a4988.sourceable-parts: has no reviewed pass."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reference row tests/fixtures/reference_inputs/interface_references.json does not validate, so it records no reviewed exact order code with current stock (the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; stepper-a4988.a4988: has no reviewed pass; stepper-a4988.switches: has no reviewed pass; stepper-a4988.motor-power: has no reviewed pass; stepper-a4988.limits: has no reviewed pass; stepper-a4988.sourceable-parts: has no reviewed pass)."
        }
    },
    "stm32-min": {
        "operating_limits": [
            {
                "part": "STM32F103C8T6",
                "source": "https://www.st.com/resource/en/datasheet/stm32f103c8.pdf",
                "limit": "2.0-3.6 V supply, 72 MHz maximum, -40 to +85 C; LCSC C8734 catalog stock 270767",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            },
            {
                "part": "X32258MSB4SI",
                "source": "https://www.lcsc.com/datasheet/C2682774.pdf",
                "limit": "8 MHz, 20 pF load, +/-10 ppm, -40 to +85 C; LCSC C2682774 catalog stock 96378",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            },
            {
                "part": "TPS54331DDAR",
                "source": "https://www.ti.com/lit/ds/symlink/tps54331.pdf",
                "limit": "3.5-28 V input, adjustable output set to 3.3 V by the recipe's 68.1k/22k divider, up to 3 A; LCSC C90761",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            },
            {
                "part": "TYPE-C-31-M-12",
                "source": "https://www.lcsc.com/datasheet/C165948.pdf",
                "limit": "20 V, 5 A, 10000 mating cycles; LCSC C165948 catalog stock 89797",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/mcu_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations stm32-package, usb-crystal-swd, boot-reset, sourceable-parts pass over reviewed part identities (STM32F103C8T6, X32258MSB4SI, TPS54331DDAR, TYPE-C-31-M-12, kicad-tl3342-button) resolved by stm32f103c8t6-minimal@1, usb-c-usb2-device@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (programming, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: STM32F103C8T6 (LCSC C8734 as recorded in the row, catalog stock 270767 verified by read-only jlcparts.lookup on the local dump, 2026-09-16); X32258MSB4SI (LCSC C2682774 as recorded in the row, catalog stock 96378 verified by read-only jlcparts.lookup on the local dump, 2026-09-16); TPS54331DDAR (LCSC C90761 as recorded in the row, catalog stock 47961 verified by read-only jlcparts.lookup on the local dump, 2026-09-16); TYPE-C-31-M-12 (LCSC C165948 as recorded in the row, catalog stock 89797 verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "rp2040-min": {
        "operating_limits": [
            {
                "part": "RP2040",
                "source": "https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf",
                "limit": "IO supply 1.8-3.3 V, 133 MHz maximum, core fed by the internal 1.1 V regulator; LCSC C2040 catalog stock 53973",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            },
            {
                "part": "W25Q16JVSS",
                "source": "https://www.winbond.com/hq/support/documentation/levelOne.jsp?__locale=en&DocNo=DA00-W25Q16JV.1",
                "limit": "16 Mbit serial flash, 2.7-3.6 V, standard/dual/quad SPI (KiCad Memory_Flash:W25Q16JVSS datasheet field)",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            },
            {
                "part": "ABM8-272-T3",
                "source": "https://abracon.com/datasheets/ABM8-272-T3.pdf",
                "limit": "12 MHz, 10 pF load, +/-30 ppm, -40 to +85 C; LCSC C20625731 catalog stock 18110",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            },
            {
                "part": "TYPE-C-31-M-12",
                "source": "https://www.lcsc.com/datasheet/C165948.pdf",
                "limit": "20 V, 5 A, 10000 mating cycles; LCSC C165948 catalog stock 89797",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/mcu_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations rp2040-package, core-support, sourceable-parts pass over reviewed part identities (RP2040, W25Q16JVSS, ABM8-272-T3, TYPE-C-31-M-12, TPS54331DDAR) resolved by rp2040-minimal@2, usb-c-usb2-device@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (castellations, programming, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: RP2040 (LCSC C2040 as recorded in the row, catalog stock 53973 verified by read-only jlcparts.lookup on the local dump, 2026-09-16); ABM8-272-T3 (LCSC C20625731 as recorded in the row, catalog stock 18110 verified by read-only jlcparts.lookup on the local dump, 2026-09-16); TYPE-C-31-M-12 (LCSC C165948 as recorded in the row, catalog stock 89797 verified by read-only jlcparts.lookup on the local dump, 2026-09-16); TPS54331DDAR (LCSC C90761 as recorded in the row, catalog stock 47961 verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "esp32-s3-sensor": {
        "operating_limits": [
            {
                "part": "ESP32-S3-WROOM-1-N8R8",
                "source": "https://www.lcsc.com/product-detail/C2913201.html",
                "limit": "3.0-3.6 V supply, -40 to +65 C, 8 MB flash / 8 MB PSRAM, on-board PCB antenna",
                "disposition": "reviewed module; the reviewed 3.3 V USB-C supply and I2C sensor interface are inside the range"
            },
            {
                "part": "BME280",
                "source": "https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme280-ds002.pdf",
                "limit": "1.71-3.6 V supply, -40 to +85 C, 0-100 %RH, I2C/SPI",
                "disposition": "reviewed temperature/humidity/pressure interface on the 3.3 V rail"
            },
            {
                "part": "TYPE-C-31-M-12",
                "source": "https://www.lcsc.com/product-detail/C165948.html",
                "limit": "20 V / 5 A, 16 contacts, 10 000 mating cycles",
                "disposition": "reviewed USB-C power/programming input"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/mcu_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations parts, usb-programming, status-led, sourceable-parts pass over reviewed part identities (ESP32-S3-WROOM-1-N8R8, BME280, TYPE-C-31-M-12, GRM188R71C104KA01D) resolved by esp32-s3-wroom-1-minimal@1, usb-c-usb2-device@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Espressif ESP32-S3-WROOM-1-N8R8 (LCSC C2913201, catalog stock 6547 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Bosch BME280 (LCSC C92489, catalog stock 10327 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Korean Hroparts TYPE-C-31-M-12 (LCSC C165948, catalog stock 89797 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "nrf52-beacon": {
        "operating_limits": [
            {
                "part": "NRF52840-QIAA-R7",
                "source": "https://docs.nordicsemi.com/r/bundle/ps_nrf52840/page/ordering_info.html",
                "limit": "1.7-5.5 V supply, 2.4 GHz Bluetooth LE radio, aQFN-73",
                "disposition": "reviewed SoC; the reviewed CR2032 3.0 V cell is inside the range (nrf52-beacon.reviewed-nrf)"
            },
            {
                "part": "H2U38D1E1B0100",
                "source": "https://www.unictron.com/wp-content/uploads/datasheet/H2U38D1E1B0100.pdf",
                "limit": "2400-2500 MHz, 50 ohm, 2 W maximum input, -40 to +85 C",
                "disposition": "reviewed chip antenna on the reviewed 2.4 GHz RF path (nrf52-beacon.rf)"
            },
            {
                "part": "TL3342F260QG",
                "source": "https://www.lcsc.com/product-detail/C2886894.html",
                "limit": "50 mA / 12 V, 100 000 cycles, -20 to +70 C",
                "disposition": "reviewed user button"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/mcu_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations reviewed-nrf, rf, user-power, sourceable-parts pass over reviewed part identities (NRF52840-QIAA-R7, H2U38D1E1B0100, BS-07-A1BJ001, TL3342F260QG) resolved by the reviewed recipe/lowerer work units; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (antenna-geometry, programming, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Nordic NRF52840-QIAA-R7 (LCSC C1851953, catalog stock 3626 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Unictron H2U38D1E1B0100 (LCSC C6569546, catalog stock 2485 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); MYOUNG BS-07-A1BJ001 CR2032 holder (LCSC C2979167, catalog stock 1556 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); E-Switch TL3342F260QG (LCSC C2886894, catalog stock 445 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "lora-node": {
        "operating_limits": [
            {
                "part": "DL-RFM95-868M",
                "source": "https://www.dreamlnk.com/en/DL-RFM95.html",
                "limit": "1.8-3.7 V supply, 820-1020 MHz, +19.5 dBm maximum output",
                "disposition": "reviewed SX1276-based radio module; 868 MHz is the reference stated engineering assumption for the unspecified band (lora-node.radio)"
            },
            {
                "part": "STM32L031K6T6",
                "source": "https://www.st.com/resource/en/datasheet/stm32l031k4.pdf",
                "limit": "1.65-3.6 V supply, 32 MHz maximum clock",
                "disposition": "reviewed MCU sharing the module 3.3 V domain"
            },
            {
                "part": "132289",
                "source": "https://www.lcsc.com/product-detail/C3172723.html",
                "limit": "50 ohm, DC-18 GHz, board-edge SMA jack for a 1.57 mm board",
                "disposition": "reviewed SMA connector on the reviewed RF path (lora-node.sma)"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/mcu_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations radio, sma, mcu-terminal, sourceable-parts pass over reviewed part identities (STM32L031K6T6, DL-RFM95-868M, 132289, WJ126V-5.0-04P-14-00A, WJ126V-5.0-02P-14-00A) resolved by the reviewed recipe/lowerer work units; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: STMicroelectronics STM32L031K6T6 (LCSC C94085, catalog stock 1595 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); DreamLNK DL-RFM95-868M (LCSC C2844472, catalog stock 1062 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Amphenol RF 132289 SMA (LCSC C3172723, catalog stock 215 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "esp32-dual-motor": {
        "operating_limits": [
            {
                "status": "unverified",
                "reason": "No reviewed operating limit is relied on for this brief yet: the reference row tests/fixtures/reference_inputs/mcu_references.json does not validate, so it records no reviewed part limit this disposition can rely on."
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "Reference row tests/fixtures/reference_inputs/mcu_references.json does not validate against its contract, so no reviewed boundary commit is evidenced: the row brief hash does not match the frozen original brief; the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; esp32-dual-motor.parts: has no valid result status; esp32-dual-motor.esp32-s3: has no valid result status; esp32-dual-motor.motor-channels: has no valid result status; esp32-dual-motor.buck: has no valid result status; esp32-dual-motor.sourceable-parts: has no reviewed pass."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reference row tests/fixtures/reference_inputs/mcu_references.json does not validate, so it records no reviewed exact order code with current stock (the row brief hash does not match the frozen original brief; the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; esp32-dual-motor.parts: has no valid result status; esp32-dual-motor.esp32-s3: has no valid result status; esp32-dual-motor.motor-channels: has no valid result status; esp32-dual-motor.buck: has no valid result status; esp32-dual-motor.sourceable-parts: has no reviewed pass)."
        }
    },
    "round-led-ring": {
        "operating_limits": [
            {
                "part": "ATTINY412-SSN",
                "source": "https://ww1.microchip.com/downloads/aemDocuments/documents/MCU08/ProductDocuments/DataSheets/ATtiny212-214-412-414-416-DataSheet-DS40002287A.pdf",
                "limit": "1.8-5.5 V supply, UPDI programming, 4 KB flash, -40 to +105 C",
                "disposition": "reviewed controller; the recipe attiny412-updi-minimal@1 supplies the reviewed UPDI header (round-led-ring.controller-power)"
            },
            {
                "part": "WS2812B-B/T",
                "source": "https://datasheet.lcsc.com/datasheet/pdf/bc8264a6d62c958a89e25ec8cc82b690.pdf?productCode=C2761795",
                "limit": "3.7-5.3 V supply, 60 mA full white, data VIH at 0.7 x VDD",
                "disposition": "reviewed LEDs; the reviewed 5 V inlet and ATtiny412 data level must satisfy VIH"
            },
            {
                "part": "S2B-PH-SM4-TB(LF)(SN)",
                "source": "https://lcsc.com/product-detail/_JST-Sales-America_S2B-PH-SM4-TB-LF-SN_JST-Sales-America-S2B-PH-SM4-TB-LF-SN_C295747.html",
                "limit": "1x02 2.00 mm PH inlet; design supply 720 mA with a 1.0 A source requirement in a 4.75-5.25 V window",
                "disposition": "reviewed power inlet for the twelve-LED ring"
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "The reference row validates its reviewed obligations, but its own unresolved coverage records that twelve-instance expansion, interior JST-PH placement and circular placement are unimplemented deterministic coverage, so the deterministic BOM/wiring stage commit is not evidenced."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Worldsemi WS2812B-B/T (LCSC C2761795, catalog stock 370560 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); JST S2B-PH-SM4-TB(LF)(SN) (LCSC C295747, catalog stock 113589 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); Microchip ATTINY412-SSN (LCSC C1337190, catalog stock 2638 matched in the catalog by the SOIC-8 AVR ATtiny412 signature, verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "rounded-c3-devboard": {
        "operating_limits": [
            {
                "status": "unverified",
                "reason": "No reviewed operating limit is relied on for this brief yet: the reference row tests/fixtures/reference_inputs/mcu_references.json does not validate, so it records no reviewed part limit this disposition can rely on."
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "Reference row tests/fixtures/reference_inputs/mcu_references.json does not validate against its contract, so no reviewed boundary commit is evidenced: the row brief hash does not match the frozen original brief; the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; rounded-c3-devboard.controller: has no valid result status; rounded-c3-devboard.edge-interfaces: has no valid result status; rounded-c3-devboard.sourceable-parts: has no reviewed pass."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reference row tests/fixtures/reference_inputs/mcu_references.json does not validate, so it records no reviewed exact order code with current stock (the row brief hash does not match the frozen original brief; the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; rounded-c3-devboard.controller: has no valid result status; rounded-c3-devboard.edge-interfaces: has no valid result status; rounded-c3-devboard.sourceable-parts: has no reviewed pass)."
        }
    },
    "chamfered-badge": {
        "operating_limits": [
            {
                "status": "unverified",
                "reason": "No reviewed operating limit is relied on for this brief yet: the reference row tests/fixtures/reference_inputs/mcu_references.json does not validate, so it records no reviewed part limit this disposition can rely on."
            }
        ],
        "feasibility": {
            "status": "not_yet_reviewed",
            "reason": "Reference row tests/fixtures/reference_inputs/mcu_references.json does not validate against its contract, so no reviewed boundary commit is evidenced: the row brief hash does not match the frozen original brief; the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; chamfered-badge.parts: has no valid result status; chamfered-badge.touch: has no valid result status; chamfered-badge.budget: has no valid result status; chamfered-badge.sourceable-parts: has no reviewed pass."
        },
        "sourceability": {
            "status": "unverified",
            "reason": "Not verified: the reference row tests/fixtures/reference_inputs/mcu_references.json does not validate, so it records no reviewed exact order code with current stock (the row brief hash does not match the frozen original brief; the row's deferred set does not match the contract's deferred obligations; the row's obligation ids do not match the contract's reference obligations; chamfered-badge.parts: has no valid result status; chamfered-badge.touch: has no valid result status; chamfered-badge.budget: has no valid result status; chamfered-badge.sourceable-parts: has no reviewed pass)."
        }
    },
    "star-ornament": {
        "operating_limits": [
            {
                "part": "ATTINY402-SSNR",
                "source": "https://jlcpcb.com/partdetail/MicrochipTech-ATTINY402SSNR/C616056",
                "limit": "1.8-5.5 V, -40..105 C, 20 MHz; LCSC C616056 (assembly stock 210 in the reviewed catalog dump)",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            },
            {
                "part": "BS-07-A1BJ001",
                "source": "https://www.lcsc.com/datasheet/C2979167.pdf",
                "limit": "CR2032 coin cell only (3.0 V nominal); removable cell supplied separately; LCSC C2979167 (stock 1556 in the reviewed catalog dump)",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            },
            {
                "part": "E6C0805WWAY1UDA(1.1T M)",
                "source": "https://www.lcsc.com/product-detail/Light-Emitting-Diodes-LED_EKINGLUX-E6C0805WWAY1UDA-1-1T-M_C6916225.html",
                "limit": "3.0 V forward at 20 mA, 2800-3200 K warm white; LCSC C6916225 (stock 6651 in the reviewed catalog dump)",
                "disposition": "reviewed limit recorded by the reference row's part_inventory; the row's reference obligations rely on this part"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/mcu_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations leds, controller-power, sourceable-parts pass over reviewed part identities (ATTINY402-SSNR, BS-07-A1BJ001, E6C0805WWAY1UDA(1.1T M)) resolved by attiny402-ssnr-updi-minimal@1, coin-cell-holder@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (outline, led-point-placement, hang-hole, budget-programming, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: ATTINY402-SSNR (LCSC C616056 as recorded in the row, catalog stock 210 verified by read-only jlcparts.lookup on the local dump, 2026-09-16); BS-07-A1BJ001 (LCSC C2979167 as recorded in the row, catalog stock 1556 verified by read-only jlcparts.lookup on the local dump, 2026-09-16); E6C0805WWAY1UDA(1.1T M) (LCSC C6916225 as recorded in the row, catalog stock 6651 verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    },
    "snowman-ornament": {
        "operating_limits": [
            {
                "part": "ATTINY402-SSNR",
                "source": "https://jlcpcb.com/partdetail/MicrochipTech-ATTINY402SSNR/C616056",
                "limit": "1.8-5.5 V, -40 to +105 C, 20 MHz, SOIC-8",
                "disposition": "reviewed controller; the reviewed attiny402-ssnr-updi-minimal@1 recipe supplies UPDI (snowman-ornament.controller-power)"
            },
            {
                "part": "BS-07-A1BJ001",
                "source": "https://www.lcsc.com/datasheet/C2979167.pdf",
                "limit": "CR2032 coin cell holder, 3.0 V nominal",
                "disposition": "reviewed cell holder supplying the reviewed 3.0 V rail"
            },
            {
                "part": "E6C0805WWAY1UDA(1.1T M)",
                "source": "https://www.lcsc.com/product-detail/Light-Emitting-Diodes-LED_EKINGLUX-E6C0805WWAY1UDA-1-1T-M_C6916225.html",
                "limit": "3.0 V forward at 20 mA, 2800-3200 K warm white",
                "disposition": "reviewed warm-white LEDs in three independently current-limited channels"
            }
        ],
        "feasibility": {
            "status": "reviewed_feasible",
            "reason": "Validated reference row tests/fixtures/reference_inputs/mcu_references.json exercises 'architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit': reference obligations leds, controller-power, sections, sourceable-parts pass over reviewed part identities (ATTINY402-SSNR, BS-07-A1BJ001, E6C0805WWAY1UDA(1.1T M)) resolved by attiny402-ssnr-updi-minimal@1, coin-cell-holder@1; pin/footprint mapping, complete-required-connections, ERC/DRC, exported artifacts and the geometry/programming gates (outline, budget-programming, pin-footprint-mapping, complete-required-connections, exported-artifacts, erc, drc, programming-when-applicable, geometry-when-applicable) are recorded as deferred to campaign fulfillment, not claimed here."
        },
        "sourceability": {
            "status": "reviewed_sourceable",
            "reason": "Reviewed sourceability: Microchip ATTINY402-SSNR (LCSC C616056, catalog stock 210 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); MYOUNG BS-07-A1BJ001 CR2032 holder (LCSC C2979167, catalog stock 1556 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16); EKINGLUX E6C0805WWAY1UDA(1.1T M) (LCSC C6916225, catalog stock 6651 recorded in the row's part_inventory sourcing_check_at_review, re-verified by read-only jlcparts.lookup on the local dump, 2026-09-16)."
        }
    }
}


# Apply the reviewed dispositions over the fail-closed placeholder defaults.
for _slug, _disposition in _REVIEWED_DISPOSITIONS.items():
    ORIGINAL_CONTRACTS[_slug].update(deepcopy(_disposition))


# Reviewed engineering assumptions per brief, taken verbatim from that brief's
# reference row (or recorded as UNREVIEWED when no row validates).  A model
# default is not an approval and an assumption is not a substitution: this
# table authorizes nothing.  Every substitution needs its own ``consent``
# record naming the approver and the changed requirement, and
# ``substitutions`` below lists only those consent-gated changes.
_REVIEWED_ASSUMPTIONS: dict[str, tuple[str, ...]] = {
    "audio-jack-buffer": (
            "Each SJ1-3533NG is used as a TS mono line input: sleeve contact S is AGND, tip contact T is the channel input, and ring contact R is deliberately NC (not shorted).",
            "Inputs are at most 2.0 Vpp sine centered about external AC ground; each channel is AC coupled with a reviewed 10 uF 0603 X5R capacitor into a 100 kOhm VREF bias path, so the nominal coupling corner is 0.16 Hz and stays far below the 20 Hz audio band even after X5R DC-bias derating at 2.5 V.",
            "The four output loads are at least 10 kOhm and are not headphones, speakers, or direct cable-capacitive loads.",
            "MCP6004 runs from +5V and AGND, with VREF=2.5V from a 10k/10k divider and bypass; all four amplifiers are unity followers about VREF.",
            "The original brief names no power inlet and the jack count is exactly four, so the reviewed reference adds one 2-pin 2.54 mm header (pin1=+5V, pin2=AGND) as the external 5 V inlet; it is not a fifth jack.",
            "The 100 kOhm bias/output resistors keep the reviewed design value; the reviewed 0603 resistor record is 10 kOhm (ERJ-3EKF1002V), so the 100 kOhm order code is recorded as a pending sourcing gap rather than substituted silently.",
    ),
    "buck-3a": (
            "RECORDED SPECIFICATION CONFLICT (unresolved): the original brief requires a 5 V input, but the TPS5430 recommended input range is 5.5-36 V and its UVLO prevents start-up until VIN reaches 5.5 V (5.3 V typical). This reference keeps the original device and the original 5 V/3.3 V/3 A specification and does NOT claim a compliant 5 V-input implementation; the row is evidence of device identity, terminals, output set-point and current rating only.",
            "Support circuit per TI datasheet SLVS632L: BOOT-to-PH 0.01 uF low-ESR bootstrap capacitor rated at least 10 V (C0805C103J5GACTU 10 nF/50 V C0G); 15 uH output inductor (FXL0630-150-M) from PH to the 3.3 V output; B340A 40 V/3 A catch diode from PH to GND; VIN decoupled with a 10 uF X7R ceramic plus a 100 nF high-frequency bypass; 220 uF low-ESR output capacitor; feedback divider R1 10.0 kOhm / R2 5.9 kOhm giving 3.291 V nominal; ENA left floating (internal 1.5 MOhm pull-up, no resistor to ground); GND pin connected to the exposed PowerPAD.",
            "The reviewed 15 uH asset is rated 3 A RMS / 4 A saturation while TI's procedure computes 3.003 A RMS and 3.31 A peak at 3 A output: the RMS rating is at the boundary (0.1%) and the peak has 21% margin. A production build should confirm the DC-bias derating or select the higher-current 15 uH equivalent (e.g. SMMS1040-150M, LCSC C149596, 6 A/7 A) - recorded as a design note, not a silent substitution.",
            "Delivered thermal-via copper, PowerPAD land, ERC/DRC, exported artifacts and the geometry gate remain deferred obligations.",
    ),
    "can-node": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "chamfered-badge": (
            "CR2032 nominal 3V drives six warm-white 0805 LEDs at 1mA each through 220ohm; firmware permits one LED at a time.",
            "Touch pads use ATtiny1614 PTC-capable PA4/PA5 and have no copper underlay or conductive overlay.",
    ),
    "daq-8ch": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "dual-rail-supply": (
            "WRA2412S-3WR2 driver-side parts follow the ReSine datasheet design reference: Cin1 10 uF, Cin2 1 uF, Lin 4.7-12 uH, Cs 10-22 uF, Lout 2.2-10 uH, Cout 100 uF typical.",
            "Pin 3 (CTRL) is left open and pin 5 (NC) is unconnected: both are required no-connects.",
            "The 24 V input terminal and the +12 V/0 V/-12 V output terminal are realized by the reviewed screw-terminal lowerer requirements.",
            "CIN1 is the ROQANG RVT2A100M0607 10 uF / 100 V SMD electrolytic (LCSC C72482, D6.3 x L7.7 mm, assembly stock 121902, retail stock 75020): the same reviewed MPN and land pattern as the DMBJ RVT2A100M0607 record (LCSC C970659), whose record has only 56 assembly units and 20 retail units and therefore fails the assembly floor and the storefront minimum buy. Recorded as an explicit sourced substitution, not a silent change of value, voltage or footprint.",
    ),
    "encoder-oled-panel": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "esp32-dual-motor": (
            "DRV8833 logic high thresholds are 2.0 V on AIN/BIN and 2.5 V on nSLEEP. ESP32 3.3 V GPIO meets both. nFAULT is open drain and each returned fault net has its own 10 kOhm pull-up to 3V3. AISEN/BISEN are grounded because the original brief gives no motor current target; this retains TI overcurrent protection but does not claim a chosen chopping current.",
            "The recipe-owned ESP32-S3 USB, BOOT, and RESET circuit remains responsible for native USB programming. GPIO4/5/6/7/15/16/17/18/8/9 retain the ten motor-control allocations (module pins 4/5/6/7/8/9/10/11/12/17); GPIO10/11 retain the two returned nFAULT inputs (module pins 18/19). GPIO0, GPIO19 and GPIO20 remain boot/native-USB pins.",
            "The 3.3 V rail is the reviewed TPS54331DDAR adjustable buck recipe: 3.5-28 V input (2S 6.0-8.4 V inside range), 3 A continuous output, 15 uH switch inductor, 10 uF input and 22 uF output capacitors, 68.1k/22k feedback for 3.3 V, 511k/100k enable divider, 100 nF bootstrap and soft-start/compensation network. Inductor saturation, capacitor voltage and thermal bounds still require independent review (recipe assertion buck_component_bounds).",
            "The reference keeps the regulated 3.3 V logic rail separate from the unregulated 2S VBAT motor rail (DRV8833 VM range 2.7-10.8 V).",
            "Reference assignment from the deterministic commit: U1/U2 DRV8833PWPR, U3 ESP32-S3-WROOM-1-N8R8, U4 TPS54331DDAR, U5 USBLC6-2SC6, J1 USB-C, J2 2S battery terminal, J3-J6 motor screw terminals, L1 15 uH; C1-C8 driver capacitors, C9-C11 ESP32 capacitors, C12-C18 buck capacitors; R1-R4 driver sense, R5/R6 ESP32 pull-ups, R7-R11 buck divider/enable, R12/R13 USB-C CC 5.1k, R14/R15 USB series 22R; SW1 RESET, SW2 BOOT.",
    ),
    "esp32-s3-sensor": (
            "ESP32-S3-WROOM-1-N8R8 is supplied from a regulated 3.3V rail rated at least 1A.",
            "BME280 uses 3.3V I2C at address 0x76; GPIO4 drives a status LED through 1k to GND.",
    ),
    "fpc-breakout": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "gpio-expander": (
            "MCP23017-E/SO runs at 3.3 V on pin 9 (VDD) with pin 10 (VSS) grounded and ~RESET (pin 18) tied to +3V3; A2:A0 (pins 17, 16, 15) are strapped to GND, selecting I2C address 0x20 so a chained board only changes its straps.",
            "One 10 kOhm pull-up pair (R1/R2, reviewed ERJ-3EKF1002V) is fitted once per board on SDA and SCL. Standard-mode I2C at 100 kHz allows at most Rp(max) = tr(max)/(0.8473 x Cb) = 1000 ns / (0.8473 x 100 pF) = 11.8 kOhm for a 100 pF bus, so the reviewed 10 kOhm part is inside the bound for two short 0.1-inch header stubs and the device pins.",
            "J1 and J2 are identical 1x04 0.1-inch headers wired to the same GND/+3V3/SDA/SCL nets, so a second identical cable chains the bus to the next board without a second pull-up set.",
            "One 100 nF X7R bypass (C1) sits at the MCP23017 VDD pin. The sixteen GPIO lines each terminate on their own 2-position 5.00 mm screw terminal (signal + ground).",
            "The reviewed MCP23017 identity relation accepts exactly MCP23017-E/SO; the vendored mcp23017t-e-ss asset (SSOP-28 MCP23017T-E/SS) is not accepted, so this reference uses the SOIC-28 order code with the KiCad standard symbol/footprint pair.",
    ),
    "hex-env-sensor": (
            "The Qwiic host provides regulated 3.3 V on pin 2 and I2C pull-ups on SDA/SCL.",
            "BME280 uses I2C mode: CSB (pin 2) is tied to VDDIO and SDO (pin 5) is tied to GND, selecting address 0x76.",
            "The power LED is green at 2 mA from +3V3; the lowerer calculates its resistor.",
    ),
    "highside-switch-10a": (
            "The switched rail is an explicit 15 V application assumption; the brief names no voltage. AONR21357 is rated 30 V VDS and +/-25 V VGS.",
            "Open-collector NPN Q2 pulls the gate low through the 1 kOhm gate series resistor; the 100 kOhm gate pull-up to the source holds the switch off when CTRL is low or floating.",
            "The 10 V source-gate zener (cathode to the source, anode to the gate) clamps |VGS| to at most the zener maximum breakdown, keeping the gate inside the +/-25 V absolute maximum.",
            "1 kOhm gate series resistance limits continuous zener current to about 4.4 mA and steady dissipation to about 45 mW of the 500 mW rating; gate turn-off through the 100 kOhm pull-up is about 283 us, acceptable for a static DC load switch.",
            "All parallel source pins 1/2/3 tie to +15V_IN and all drain pins 5/6/7/8 plus exposed pad 9 tie to +15V_LOAD.",
    ),
    "led-cc-driver": (
            "USB-C 5 V VBUS is the input; the AL8860 VIN range 4.5-40 V includes it, and the datasheet requires a 10 uF or larger X7R decoupling capacitor at VIN (two 10 uF 25 V X7R 1210 parts are fitted, D6/F4).",
            "RSET of 0.100 Ohm between VIN and SET sets the nominal average LED current to 0.1 V / 0.100 Ohm = 1.0 A (AL8860 datasheet pin table: IOUT_NOM = 0.1/Rs; VSENSE 96/100/104 mV measured on SET with respect to VIN).",
            "CTRL is left floating for normal full-current operation, exactly as the datasheet pin description requires; the 22 uH shielded inductor and the SS34 Schottky freewheel diode complete the buck current path from SW.",
            "The external LED mounts on the reviewed 2-position screw terminal between LED_ANODE (SET side) and LED_CATHODE (inductor side).",
            "Heatsink copper under the exposed pad with thermal vias remains a delivered-geometry obligation, not claimed by this reference row.",
    ),
    "lora-node": (
            "The original brief leaves frequency and supply source open. This reference selects DreamLNK DL-RFM95-868M, a genuine Semtech-SX1276 868 MHz module, and an externally regulated 3.3 V input at J4; regional authorization and antenna certification are unverified.",
            "The sensor terminal is a 3.3 V I2C interface: J2.1=3V3, J2.2=GND, J2.3=SDA, J2.4=SCL. It is not a field-power or high-voltage interface.",
            "STM32L031K6T6 uses the internal HSI clock; PC14/PC15 have no fitted LSE crystal. BOOT0 is held low by R1, NRST is pulled high by R2 and exposed on SWD.",
            "DL-RFM95-868M contains the radio core; SX1276 remains the named radio family obligation and is not used as the module MPN.",
    ),
    "nrf52-beacon": (
            "nRF52840-QIAA-R7 is retained exactly; R7 is the reviewed reel order code.",
            "CR2032 is nominal 3V. RF matching and antenna copper keepout follow Nordic reference layout and require eventual geometry evidence.",
    ),
    "proto-shield": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "r2r-dac": (
            "Logic inputs are 0 V/3.3 V CMOS signals on pins 1-8 of a 0.1 inch header.",
            "The same header supplies +3.3 V logic reference (pin 9), regulated +5 V analog supply (pin 10), and ground (pin 11).",
            "The buffered analog output drives a high-impedance load of at least 10 kohm; a full-scale 3.3 V ladder output is within the 5 V MCP6001 supply range.",
    ),
    "rc-lowpass-bnc": (
            "3296W-1-103LF is wired as a rheostat: pin 1 is the filter input and the tied wiper pins 2/3 are the filtered output, so an open wiper cannot leave the output floating (Bourns 3296 datasheet page 1 wiring diagram: wiper = 2, CCW = 1, CW = 3).",
            "The reviewed lowerer fixes C = 10 nF C0G/NP0 (C0805C103J5GACTU) and R_max = 10 kohm, so the minimum cutoff is f_min = 1/(2*pi*R_max*C) = 1591.5 Hz.",
            "Sweeping the 25-turn trimmer down toward its minimum resistance raises the cutoff above 1.5915 kHz; the brief fixes no upper bound, and the 3296W absolute-minimum-resistance limit (1 % of nominal or 2 ohm, whichever is greater = 100 ohm here) bounds the practical sweep at about 159 kHz.",
            "The filter is a high-impedance passive RC: it does not claim a matched 50 ohm transfer, because the source and load impedances were never specified.",
    ),
    "relay-quad": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "round-led-ring": (
            "J1 (S2B-PH-SM4-TB(LF)(SN)) is an interior side-entry 2-pin JST PH inlet: pin 1 = regulated 5 V, pin 2 = GND. Pads 3 and 4 are the mechanical mounting tabs and are explicitly not electrical nets.",
            "The inlet carries the 12-pixel full-white budget of 12 x 60 mA = 720 mA, so the external source and harness must be rated for at least 1 A continuous; the vendored JST PH asset records a 1.0 A design source requirement and a 4.75-5.25 V input window.",
            "ATtiny412 (ATTINY412-SSN) runs from the same 5 V rail, well inside its 1.8-5.5 V operating range, and is programmed through UPDI on PA0 (pin 7).",
            "The twelve WS2812B pixels are one recipe instance: ws2812-output@1 with quantity 12 cascades D1..D12 through the recipe's own 330 ohm series resistor and per-pixel 100 nF decoupling, leaving D12.DOUT as a no-connect.",
            "No edge connector exists anywhere in the reference: the JST PH inlet and the UPDI header are interior parts of a 60 mm circular outline, and the architecture declares no edge: signal.",
    ),
    "rounded-c3-devboard": (
            "Reference chooses 3mm corner radius, USB-C on bottom edge, actual 2x10 2.54mm header on opposite top edge.",
            "GPIO9 is the ESP32-C3 boot strap and is not exposed; BOOT/RESET controls are recipe-internal nets, not header pins. The 2x10 header repeats 3V3/GND/VBUS on five positions to fill all 20 positions.",
            "The 3.3 V rail is owned by the reviewed TLV62569DBVR 2 A buck recipe (tlv62569-3v3@1, 2.5-5.5 V input, power_transfer VIN->SW) rather than the ME6211 linear regulator whose reviewed record carries no source-to-load transfer contract; 5 V VBUS input and the ESP32-C3 3.3 V/0.5 A load are inside reviewed limits.",
    ),
    "rp2040-min": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "rs485-terminal": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "servo-driver-16": (
            "External 5 V servo power is 8 A maximum at the screw terminal; the reviewed reference assumes 0.5 A average per channel and 16 x 0.5 A = 8 A of terminal/copper distribution, never through the PCA9685.",
            "PCA9685 logic runs at 3.3 V; the PWM outputs are 5.5 V tolerant open-drain-style drivers, so a 5 V servo signal is compatible without a level shifter, but a servo whose logic high requirement exceeds 3.3 V must be reviewed.",
            "The sixteen 3-pin headers are the deterministic connector-bank@1 connector bank (1 pin per channel group of GND / +5V_SERVO / PWMn); their physical hardware is the vendored BOOMELE 2.54 mm breakaway header cut to 1x03, rated 3 A per pin at 2.54 mm pitch.",
            "The power screw terminal is the reviewed WJ126V-5.0-02P-14-00A (250 V / 18 A), which covers the 8 A design maximum with margin.",
            "The PCA9685 is the vendored PCA9685PW,118 TSSOP-28 part; the acceptance contract names the device as PCA9685, and the reviewed identity table has no PCA9685 family relation yet, so this reference records the exact device designation (see the pca9685 obligation).",
    ),
    "snowman-ornament": (
            "CR2032 nominal 3V drives one warm-white LED at a time at 1mA maximum through 220ohm; firmware serializes illumination.",
            "ATTINY402-SSNR is the reviewed tape-and-reel, same-package and same-grade ATtiny402 selection; no other MCU substitution is authorized.",
    ),
    "speaker-crossover": (
            "UNREVIEWED: no validating reference row (reference fixture defers exactly speaker-crossover.complete-required-connections, speaker-crossover.drc, speaker-crossover.erc, speaker-crossover.exported-artifacts, speaker-crossover.geometry-when-applicable, speaker-crossover.pin-footprint-mapping, speaker-crossover.programming-when-applicable).",
    ),
    "star-ornament": (
            "CR2032 nominal 3V drives one warm-white LED at a time at 1mA maximum through 220ohm; firmware serializes illumination.",
            "ATTINY402-SSN is retained exactly; no MCU substitution is authorized.",
            "The reviewed orderable tinyAVR-0 implementation is ATTINY402-SSNR (SOIC-8, -40..105 C, 1.8-5.5 V, tape-and-reel; LCSC C616056, reviewed assembly stock 210). It is the same device/package/grade as ATTINY402-SSN (C1339884), whose refreshed storefront evidence showed only retail stock 19 with a 5-piece order limit.",
    ),
    "stepper-a4988": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "stm32-min": (
            "USB-C is USB 2.0 device power/data; no PD current target is implied.",
            "8 MHz HSE uses OSC_IN/OSC_OUT with load capacitors selected to the crystal CL.",
            "The 3.3 V rail is a reviewed TPS54331DDAR adjustable buck set to 3.3 V from the 5 V USB VBUS; its inductor, feedback divider and enable network are owned by the recipe.",
    ),
    "thermocouple-amp": (
            "The reviewed MAX31855KASA+ record requires 3.0-3.6 V, so the board is a 3.3 V host-interface design and the SPI levels are 3.3 V CMOS.",
            "The K-type input lands on the reviewed 2-position 5.00 mm screw terminal (WJ126V-5.0-02P-14-00A): T+ on J2.1 to U1.3 and T- on J2.2 to U1.2, per the MAX31855 cold-junction compensation pinout.",
            "The SPI output header (stock 2.54 mm 1x05) carries +3V3 (pin 1), GND (pin 2), SCK (pin 3), CS (pin 4) and SO (pin 5); CS is active low and driven by the host.",
            "VDD decoupling is the reviewed 100 nF 0603 position (explicit-decoupling@1) from +3V3 to GND; U1 pin 8 is the device NO-CONNECT and stays unconnected.",
    ),
    "usb-a-power-splitter": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "usb-c-full-breakout": (
            "No engineering assumptions are recorded for this brief.",
    ),
    "usb-pd-trigger": (
            "UNREVIEWED: no validating reference row (is unverified).",
    ),
}


for _slug, _assumptions in _REVIEWED_ASSUMPTIONS.items():
    ORIGINAL_CONTRACTS[_slug]["assumptions"] = list(_assumptions)
    ORIGINAL_CONTRACTS[_slug].setdefault("substitutions", [])
# The only approved substitution in the corpus is the user's 5 V device change;
# the original contract keeps an empty ledger because nothing was approved for it.

ORIGINAL_CONTRACTS["buck-3a"]["feasibility"] = {
    "status": "specification_conflict",
    "reason": "The original TPS5430 requirement specifies 5.5–36 V recommended VIN; the original brief requires 5 V input.",
}
ORIGINAL_CONTRACTS["buck-3a"]["operating_limits"] = [{
    "part": "TPS5430",
    "source": "https://www.ti.com/lit/ds/symlink/tps5430.pdf",
    "limit": "recommended VIN 5.5–36 V",
    "disposition": "original 5 V request is outside range",
}]
ORIGINAL_CONTRACTS["speaker-crossover"]["sourceability"] = {
    "status": "sourcing_blocked",
    "reason": "The original air-core inductor requirement is preserved. No eligible stocked source is recorded under the strict sourcing policy; no consigned stock or external policy extension is authorized.",
}
APPROVED_5V_DEVICE_CONTRACTS = deepcopy(ORIGINAL_CONTRACTS)
APPROVED_5V_DEVICE_CONTRACTS["buck-3a"]["substitutions"] = [{
    "approved_by": "user", "approved_on": "2026-09-16",
    "change": "Retain 5 V input and replace the original TPS5430 requirement with a reviewed 5 V-capable 3 A buck device.",
    "scope": "Approval is for the 5 V/device-category change only; exact MPN, source, compensation, layout, thermal design, and delivered-artifact evidence remain independently required.",
}]

for _contract_data in APPROVED_5V_DEVICE_CONTRACTS.values():
    _contract_data["version"] = APPROVED_5V_DEVICE_CORPUS_VERSION
_buck = APPROVED_5V_DEVICE_CONTRACTS["buck-3a"]
_buck["version"] = APPROVED_5V_DEVICE_CORPUS_VERSION
_buck["feasibility"] = {
    "status": "reviewed_feasible",
    "reason": "Engineer-selected TPS54331DDAR has a manufacturer-reviewed 3.5–28 V operating range covering the retained 5 V to 3.3 V/3 A requirement; compensation and board thermal verification remain per-run obligations.",
}
_buck["sourceability"] = {
    "status": "reviewed_sourceable",
    "reason": "TPS54331DDAR is an engineer-selected reviewed candidate; current JLC/retail qualification is required from the committed run receipt.",
}
_buck["consent"] = [{
    "approved_by": "user", "approved_on": "2026-09-16",
    "change": "Retain 5 V input and replace the original TPS5430 requirement with a reviewed 5 V-capable 3 A buck device.",
    "scope": "Approval is for the 5 V/device-category change only; exact MPN, source, compensation, layout, thermal design, and delivered-artifact evidence remain independently required.",
}]
_buck["operating_limits"] = [{
    "part": "TPS5430", "source": "https://www.ti.com/lit/ds/symlink/tps5430.pdf", "limit": "recommended VIN 5.5–36 V", "disposition": "original 5 V request is outside range",
}, {
    "part": "TPS54331DDAR", "source": "https://www.ti.com/lit/ds/symlink/tps54331.pdf", "limit": "VIN 3.5–28 V; VOUT adjustable down to 0.8 V; integrated switch supports up to 3 A continuous", "disposition": "manufacturer operating range covers 5 V to 3.3 V/3 A; datasheet thermal data still requires board-specific review",
}]
for obligation in _buck["obligations"]:
    if obligation["id"] == "buck-3a.converter":
        obligation["statement"] = "A reviewed 5 V-capable 3 A buck device and support circuit are evidenced for the retained 5 V to 3.3 V/3 A requirement."
        obligation["check"] = {"kind": "part_identity", "identity": "TPS54331DDAR"}
_buck["execution_brief"] = (
    "A 5 V to 3.3 V 3 A buck converter board using a reviewed 5 V-capable 3 A "
    "buck device, with input/output screw terminals and a thermal-via copper pour."
)
CONTRACTS_BY_VERSION = {
    ORIGINAL_CORPUS_VERSION: ORIGINAL_CONTRACTS,
    APPROVED_5V_DEVICE_CORPUS_VERSION: APPROVED_5V_DEVICE_CONTRACTS,
}


def contracts(version: str = ORIGINAL_CORPUS_VERSION) -> dict[str, dict[str, Any]]:
    """Return a copy so evaluators cannot mutate the published contract."""
    try:
        return deepcopy(CONTRACTS_BY_VERSION[version])
    except KeyError as exc:
        raise ValueError(f"unknown acceptance contract version: {version}") from exc

def contract_for(slug: str, version: str = ORIGINAL_CORPUS_VERSION) -> dict[str, Any]:
    try:
        return deepcopy(CONTRACTS_BY_VERSION[version][slug])
    except KeyError as exc:
        raise ValueError(f"unknown contract version or slug: {version}/{slug}") from exc


def corpus_identity(version: str = ORIGINAL_CORPUS_VERSION) -> list[dict[str, Any]]:
    """Ordered original identities; revised contracts never rewrite their briefs."""
    return [{"index": c["index"], "slug": c["slug"], "brief": c["original_brief"], "version": version}
            for c in contracts(version).values()]


def validate_contracts() -> list[str]:
    """Validate the static contract inventory without treating it as fulfillment."""
    errors: list[str] = []
    original_slugs = [entry["slug"] for entry in BENCHMARK_PROMPTS]
    for version, inventory in CONTRACTS_BY_VERSION.items():
        if list(inventory) != original_slugs or len(inventory) != 34:
            errors.append(f"{version}: identities do not exactly preserve the original 34 briefs")
        for slug, contract in inventory.items():
            if not contract["obligations"]:
                errors.append(f"{version}/{slug}: no mandatory obligations")
            if contract["feasibility"]["status"] not in FEASIBILITY_STATUSES:
                errors.append(f"{version}/{slug}: invalid feasibility disposition")
            if contract["version"] != version:
                errors.append(f"{version}/{slug}: embedded contract version differs")
            obligation_ids = [obligation["id"] for obligation in contract["obligations"]]
            if len(obligation_ids) != len(set(obligation_ids)):
                errors.append(f"{version}/{slug}: duplicate mandatory obligation ids")
            if any(not obligation["check"].get("kind") for obligation in contract["obligations"]):
                errors.append(f"{version}/{slug}: obligation without evidence check")
    if ORIGINAL_CONTRACTS["buck-3a"]["feasibility"]["status"] != "specification_conflict":
        errors.append("original buck contract must retain its documented specification conflict")
    return errors
