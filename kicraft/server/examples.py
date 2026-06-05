"""Curated example design briefs for the web app's onboarding.

Single source of truth for prompt inspiration, reused three ways on the landing
page (``web.index``): the animated cycling placeholder types through
``EXAMPLE_PROMPTS``, the "Surprise me" button draws a random one from it, and the
clickable chips drop a ``CHIP_PROMPTS`` entry into the box.

The list is deliberately ambitious-but-proven: every entry is shaped like a design
the web pipeline completes well (ESP32-S3 sensor boards, USB-C power, LED drivers,
discrete sensor breakouts), so a bold ask ends in a finished board rather than a
stalled run. It is plain data so it stays easy to curate.
"""
from __future__ import annotations

# Full briefs the animated placeholder cycles through and "Surprise me" draws from.
EXAMPLE_PROMPTS = [
    "An ESP32-S3 plant monitor: soil moisture + BME280 temp/humidity, a small OLED, USB-C power.",
    "A USB-C PD trigger board that outputs a switch-selectable 9 V, 12 V, or 20 V.",
    "An 8-channel addressable WS2812 LED driver running off 5 V USB-C.",
    "A BMP280 barometric weather sensor on a Qwiic/STEMMA bus, USB-C.",
    "A USB-C rechargeable 18650 flashlight: boost driver, three white LEDs, no microcontroller.",
    "An ESP32-S3 automatic cat feeder: stepper-driven auger, load-cell food sensor, USB-C.",
    "A bench breakout: USB-C in, regulated 3.3 V and 5 V rails with status LEDs.",
    "An ESP32-S3 environmental logger: BME280 + ambient light sensor + microSD, USB-C.",
    "A motion-activated USB-C night light with a PIR sensor and warm-white LEDs.",
]

# Short chip label -> full brief the chip drops into the textarea.
CHIP_PROMPTS = [
    {"label": "ESP32 sensor node",
     "prompt": "An ESP32-S3 environmental sensor node: BME280 temp/humidity/pressure, USB-C, status LED."},
    {"label": "USB-C charger",
     "prompt": "A USB-C 18650 Li-ion charger board with charge-status LEDs and battery protection."},
    {"label": "LED driver",
     "prompt": "An 8-channel addressable WS2812 LED driver running off 5 V USB-C."},
]
