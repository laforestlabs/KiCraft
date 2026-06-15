"""Diverse benchmark brief set for tuning-corpus synthesis (Phase 0).

The existing ``kicraft.server.examples.EXAMPLE_PROMPTS`` is curated for *function*
variety but is topologically narrow (mostly ESP32-S3 / USB-C sensor boards), so a
default tuned on it would overfit those layouts. This set instead maximizes the
**placement/routing** stress dimensions that the tuner actually cares about:
part count, connector density + edge constraints, fine-pitch IC escape, RF
keepouts, power/thermal planes, through-hole vs SMT mix, and hierarchy depth.

Every brief is still "ambitious-but-proven" — shaped like a design the pipeline
completes — so Phase 0 synthesis ends in a finished board, not a stalled run.

Synthesis is LLM-costed and should run ONCE on the cloud box; freeze the results
with ``kicraft.tuning.corpus.freeze_corpus`` (CLI: ``corpus-freeze``) so the tuner
re-runs place+route on them at $0. See the package README for the Phase 0 flow.
"""
from __future__ import annotations

# archetype -> the placement/routing trait it exercises (for coverage reporting)
ARCHETYPE_TRAITS = {
    "single_passive": "tiny, no-MCU, few nets, analog/THT connectors",
    "usb_c_connector": "USB-C + dense/edge connectors, opening-direction constraints",
    "fine_pitch": "QFN/LQFP/FPC fine-pitch escape routing",
    "rf_antenna": "RF section + antenna keepout zones",
    "power_thermal": "high-current planes, thermal vias, heatsink pads",
    "mixed_tht_smt": "through-hole + SMT on both sides, mounting holes",
    "hi_pin_hierarchical": "high pin-count MCU, multi-leaf hierarchy",
    "connector_dense_io": "connector-heavy I/O, repeated edge headers",
}

# Each entry: a unique kebab slug, its archetype, and the one-line brief.
BENCHMARK_PROMPTS: list[dict[str, str]] = [
    # --- single-board passive / no-MCU (small, analog, THT) ----------------
    {"slug": "rc-lowpass-bnc", "archetype": "single_passive",
     "brief": "A passive RC low-pass filter breakout with two BNC connectors and a cutoff-setting trim pot, no microcontroller."},
    {"slug": "r2r-dac", "archetype": "single_passive",
     "brief": "An R-2R resistor-ladder DAC breakout: eight logic inputs on a header into an op-amp buffered analog output, no microcontroller."},
    {"slug": "thermocouple-amp", "archetype": "single_passive",
     "brief": "A K-type thermocouple amplifier board (MAX31855) with a screw-terminal input and an SPI header output."},
    {"slug": "speaker-crossover", "archetype": "single_passive",
     "brief": "A passive two-way speaker crossover: an air-core inductor, film capacitors, and binding-post terminals, no active parts."},

    # --- USB-C / dense connectors / edge constraints -----------------------
    {"slug": "usb-pd-trigger", "archetype": "usb_c_connector",
     "brief": "A USB-C PD trigger board that outputs a switch-selectable 9 V, 12 V, or 20 V."},
    {"slug": "usb-c-full-breakout", "archetype": "usb_c_connector",
     "brief": "A USB-C receptacle breakout exposing VBUS, GND, the CC and SBU pins, and the high-speed pairs to 0.1-inch headers."},
    {"slug": "usb-a-power-splitter", "archetype": "usb_c_connector",
     "brief": "A USB-C input to dual USB-A host-power splitter with per-port current limiting and status LEDs."},
    {"slug": "rs485-terminal", "archetype": "usb_c_connector",
     "brief": "An isolated RS-485 transceiver board (MAX485) with screw-terminal A/B/GND lines and a DE/RE jumper."},

    # --- fine-pitch IC escape ---------------------------------------------
    {"slug": "stm32-min", "archetype": "fine_pitch",
     "brief": "A minimal STM32F103 development board (LQFP-48) with USB-C, an 8 MHz crystal, an SWD header, and boot/reset buttons."},
    {"slug": "rp2040-min", "archetype": "fine_pitch",
     "brief": "A minimal RP2040 board (QFN-56) with QSPI flash, USB-C, a 12 MHz crystal, and castellated GPIO."},
    {"slug": "fpc-breakout", "archetype": "fine_pitch",
     "brief": "A 24-pin 0.5 mm-pitch FPC/FFC connector breakout to a 0.1-inch header row."},

    # --- RF / antenna keepout ---------------------------------------------
    {"slug": "esp32-s3-sensor", "archetype": "rf_antenna",
     "brief": "An ESP32-S3 environmental sensor node: BME280 temp/humidity/pressure, USB-C, and a status LED."},
    {"slug": "nrf52-beacon", "archetype": "rf_antenna",
     "brief": "An nRF52840 BLE beacon with an onboard chip antenna, a coin-cell holder, and a single user button."},
    {"slug": "lora-node", "archetype": "rf_antenna",
     "brief": "A LoRa node: an SX1276 module with an SMA antenna connector, an STM32L0 MCU, and a screw-terminal sensor input."},

    # --- power / thermal ---------------------------------------------------
    {"slug": "buck-3a", "archetype": "power_thermal",
     "brief": "A 5 V to 3.3 V 3 A buck converter board (TPS5430) with input/output screw terminals and a thermal-via copper pour."},
    {"slug": "highside-switch-10a", "archetype": "power_thermal",
     "brief": "A high-side load switch: a logic-controlled P-channel MOSFET driving a 10 A load on screw terminals, with a thermal pad."},
    {"slug": "led-cc-driver", "archetype": "power_thermal",
     "brief": "A 1 A constant-current driver for a single power LED, USB-C input, with a heatsink copper area, no microcontroller."},
    {"slug": "dual-rail-supply", "archetype": "power_thermal",
     "brief": "A plus/minus 12 V dual-rail supply from a 24 V input using a DC-DC converter, output on screw terminals."},

    # --- mixed through-hole + SMT / mechanical -----------------------------
    {"slug": "relay-quad", "archetype": "mixed_tht_smt",
     "brief": "A four-channel relay board: through-hole relays driven by an SMT ULN2003, opto-isolated inputs, and screw-terminal outputs."},
    {"slug": "encoder-oled-panel", "archetype": "mixed_tht_smt",
     "brief": "A front-panel board: a through-hole rotary encoder with push button, an SMT I2C OLED, three buttons, and four mounting holes."},
    {"slug": "proto-shield", "archetype": "mixed_tht_smt",
     "brief": "An Arduino-Uno-format prototyping shield with stacking through-hole headers and an onboard SMT 3.3 V regulator."},

    # --- high pin-count MCU / hierarchical ---------------------------------
    {"slug": "esp32-dual-motor", "archetype": "hi_pin_hierarchical",
     "brief": "An ESP32-S3 robot controller with two DRV8833 motor drivers, a buck regulator from a 2S battery, and motor screw terminals."},
    {"slug": "can-node", "archetype": "hi_pin_hierarchical",
     "brief": "A CAN bus node: an STM32 MCU, an SN65HVD230 transceiver, a DB9 connector, and a switchable bus terminator."},
    {"slug": "daq-8ch", "archetype": "hi_pin_hierarchical",
     "brief": "An eight-channel data-acquisition board: an MCU reading two ADS1115 ADCs over I2C, inputs on screw terminals, USB-C."},

    # --- connector-heavy / dense I/O ---------------------------------------
    {"slug": "gpio-expander", "archetype": "connector_dense_io",
     "brief": "An I2C GPIO expander board: an MCP23017 with all sixteen lines on screw terminals and a chainable I2C header."},
    {"slug": "servo-driver-16", "archetype": "connector_dense_io",
     "brief": "A 16-channel servo driver: a PCA9685 with sixteen 3-pin servo headers along the board edge and a power screw terminal."},
    {"slug": "stepper-a4988", "archetype": "connector_dense_io",
     "brief": "A stepper motor controller using an A4988 driver, microstep-select DIP switches, a motor connector, and a 12 V screw terminal."},
    {"slug": "audio-jack-buffer", "archetype": "connector_dense_io",
     "brief": "An audio breakout: four 3.5 mm jacks along the edge into an op-amp unity-gain buffer, no microcontroller."},
]


def briefs() -> list[str]:
    """Plain brief strings (drop-in for EXAMPLE_PROMPTS-style synthesis driving)."""
    return [e["brief"] for e in BENCHMARK_PROMPTS]


def by_archetype() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for e in BENCHMARK_PROMPTS:
        out.setdefault(e["archetype"], []).append(e)
    return out


def coverage() -> dict[str, int]:
    """Brief count per archetype (every trait should be represented)."""
    return {k: len(v) for k, v in by_archetype().items()}
